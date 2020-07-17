import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default='pix2pix', help="model to learn")
parser.add_argument("dataset_name", type=str, default='facades', help="number of epochs of training")
opt = parser.parse_args()

PATH = '%s' %opt.model
if opt.model == 'DRIT':
    PATH = os.path.join(PATH, 'src/')


if opt.model != 'DRIT':
    os.system('python3 %s/%s.py --dataset_name %s' %(opt.model, opt.model, opt.dataset_name))
elif opt.model == 'DRIT':
    if opt.dataset_name == 'yosimite':
        os.system('python3 train.py --dataroot ../datasets/yosemite --name yosemite')
        os.system('tensorboard --logdir ../logs/yosemite')
    elif opt.dataset_name == 'portrait':
        os.system('python3 train.py --dataroot ../datasets/portrait --name portrait --concat 0')
        os.system('tensorboard --logdir ../logs/portrait')




