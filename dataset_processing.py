import os
import sys
import itertools
import numpy as np
import cv2
import torch

from PIL import Image
from skimage.transform import SimilarityTransform
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace


def crop(image, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped


class FaceDetector:
    """
    FaceDetector class implementation: can extract aligned
    faces from images. Uses pretrained ResNet-50.
    """
    def __init__(self, device="cuda", confidence_threshold=0.8):
        self.net = RetinaFace(cfg=cfg, phase="test").to(device).eval()
        self.decode_param_cache = {}
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.cfg = cfg = cfg_re50
        self.variance = cfg["variance"]
        cfg["pretrain"] = False

    def detect(self, image):
        device = self.device
        prior_data, scale, scale1 = self.decode_params(*image.shape[:2])

        image = np.float32(image)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(device, dtype=torch.float32)

        loc, conf, landmarks = self.net(image)
        loc = loc.cpu()
        conf = conf.cpu()
        landmarks = landmarks.cpu()

        boxes = decode(loc.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]
        landmarks = decode_landm(landmarks.squeeze(0), prior_data, self.variance)
        landmarks = landmarks * scale1
        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]

        return boxes, landmarks

    def decode_params(self, height, width):
        cache_key = (height, width)
        try:
            return self.decode_param_cache[cache_key]
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()
            prior_data = priors.data
            scale = torch.Tensor([width, height] * 2)
            scale1 = torch.Tensor([width, height] * 5)

            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = result
            return result

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))


def _main():
    torch.set_grad_enabled(False)
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    reader = cv2.VideoCapture(input_file)
    face_detector = FaceDetector()
    face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")

    for idx in itertools.count():
        success, image = reader.read()
        if not success:
            break
        boxes, landms = face_detector.detect(image)
        if boxes.shape[0] == 0:
            continue

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        max_face_idx = areas.argmax()
        landm = landms[max_face_idx]
        landmarks = landm.numpy().reshape(5, 2).astype(np.int)
        image = crop(image, landmarks, image_size=300)
        aligned = Image.fromarray(image[:, :, ::-1])
        out_path = os.path.join(output_dir, "%03d.jpg" % idx)
        aligned.save(out_path)


if __name__ == '__main__':
    _main()
