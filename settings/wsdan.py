GPU = '0'
learning_rate = 1e-3
batch_size = 15
workers = 4
epochs = 40
net = 'xception'
image_size = (300, 300)
num_attentions = 8
beta = 5e-2

tag = 'dfdc'
pretrained = '../output/xception/xception.pth'
save_dir = '../output/wsdan/'
model_name = 'model.ckpt'
log_name = 'train.log'
ckpt = False
datapath = "/mnt/ssd0/dfdc"
