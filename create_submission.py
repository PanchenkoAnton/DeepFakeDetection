import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from PIL import Image
from torchvision import transforms as T
from dataset_processing import crop, FaceDetector
from models import WSDAN, xception


class DFDCLoader:
    def __init__(self, video_dir, face_detector, transform=None,
                 batch_size=25, frame_skip=9, face_limit=25):
        self.video_dir = video_dir
        self.file_list = sorted(f for f in os.listdir(video_dir)
                                if f.endswith(".mp4"))
        self.transform = transform
        self.face_detector = face_detector
        self.record = defaultdict(list)
        self.score = defaultdict(lambda: 0.5)
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.face_limit = face_limit
        self.feedback_queue = []

    def iter_one_face(self):
        for fname in self.file_list:
            path = os.path.join(self.video_dir, fname)
            reader = cv2.VideoCapture(path)
            face_count = 0
            while True:
                for _ in range(self.frame_skip):
                    reader.grab()
                success, img = reader.read()
                if not success:
                    break
                boxes, landmarks = self.face_detector.detect(img)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                landmarks = landmarks[order]

                landmarks = landmarks.numpy().reshape(5, 2).astype(np.int)
                img = crop(img, landmarks, image_size=300)
                aligned = Image.fromarray(img[:, :, ::-1])
                if self.transform:
                    aligned = self.transform(aligned)
                yield fname, aligned

                face_count += 1
                if face_count == self.face_limit:
                    break
            reader.release()

    def __iter__(self):
        self.record.clear()
        self.feedback_queue.clear()
        batch_buf = []
        batch_count = 0
        t0 = time.time()

        for fname, face in self.iter_one_face():
            self.feedback_queue.append(fname)
            batch_buf.append(face)
            if len(batch_buf) == self.batch_size:
                yield torch.stack(batch_buf)
                batch_count += 1
                batch_buf.clear()
                if batch_count % 10 == 0:
                    elapsed = 1000 * (time.time() - t0)
                    print("T: %.2f ms / batch" % (elapsed / batch_count))
        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def feedback(self, pred):
        accessed = set()
        for score in pred:
            fname = self.feedback_queue.pop(0)
            accessed.add(fname)
            self.record[fname].append(score)
        for fname in sorted(accessed):
            self.score[fname] = np.mean(self.record[fname])
            print("[%s] %.6f" % (fname, self.score[fname]))


def _main():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    test_dir = "../input/deepfake-detection-challenge/test_videos"
    csv_path = "../input/deepfake-detection-challenge/sample_submission.csv"
    face_detector = FaceDetector()
    face_detector.load_checkpoint("../input/dfdc-pretrained-2/RetinaFace-Resnet50-fixed.pth")
    loader = DFDCLoader(test_dir, face_detector, T.ToTensor())

    model1 = xception(num_classes=2, pretrained=False)
    ckpt = torch.load("../input/dfdc-pretrained-2/xception-hg-2.pth")
    model1.load_state_dict(ckpt["state_dict"])
    model1 = model1.cuda()
    model1.eval()

    model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).cuda()
    ckpt = torch.load("../input/dfdc-pretrained-2/ckpt_x.pth")
    model2.load_state_dict(ckpt["state_dict"])
    model2.eval()

    zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cuda()
    zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cuda()

    for batch in loader:
        batch = batch.cuda(non_blocking=True)
        m1 = F.interpolate(batch, size=299, mode="bilinear")
        m1.sub_(0.5).mul_(2.0)
        m1 = model1(m1).softmax(-1)[:, 1].cpu().numpy()

        m2 = (batch - zhq_nm_avg) / zhq_nm_std
        m2, _, _ = model2(m2)
        m2 = m2.softmax(-1)[:, 1].cpu().numpy()

        prediction = 0.25 * m1 + 0.75 * m2
        loader.feedback(prediction)

    with open(csv_path) as fin, open("submission.csv", "w") as fout:
        fout.write(next(fin))
        for line in fin:
            fname = line.split(",", 1)[0]
            pred = loader.score[fname]
            print("%s,%.6f" % (fname, pred), file=fout)


if __name__ == "__main__":
    _main()
