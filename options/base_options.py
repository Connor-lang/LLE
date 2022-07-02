import argparse
import os
from xml.etree.ElementInclude import default_loader
import torch 
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_dir', required=True, help='path to dataset')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='crop to this size')
        self.parser.add_argument('--patchSize', type=int, default=64, help='crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='number of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='number of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filter channels in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g., 0, 1 , 2; use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--nThreads', type=int, default=4, help='number of threads in dataloader')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='model checkpoints saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if false, shuffle dataloader')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='maximum number of samples allowed per dataset')
        self.parser.add_argument('--resize_or_crop', type=str, default='crop', help='[resize_and_crop|crop|scale_width|scale_width_and_crop|no]')
        self.parser.add_argument('--no_flip', action='store_true', help='if true, do not flip the image for data augmentation')
        self.parser.add_argument('--use_mse', action='store_true', help='MSELoss')
        self.parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
        self.parser.add_argument('--use_ragan', action='store_true', help='use ragan')
        self.parser.add_argument('--vgg', type=float, default=1.0, help='use perceptrual loss')
        self.parser.add_argument('--vgg_choose', type=str, default='relu5_3', help='choose layer for vgg')
        self.parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
        self.parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        self.parser.add_argument('--n_patch', type=int, default=0, help='choose the number of crops for patch discriminator')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--high_times', type=int, default=400, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
        self.parser.add_argument('--new_lr', action='store_true', help='tanh')
        self.parser.add_argument('--lighten', action='store_true', help='normalize attention map')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            print(f"GPU_IDS: {self.opt.gpu_ids[0]}")
            # torch.cuda.set_device(self.opt.gpu_ids[0])
        
        args = vars(self.opt)

        print('----------Options----------')
        for k, v in sorted(args.items()):
            print(f'{str(k)}: {str(v)}')
        print('-----------End----------')

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('----------Options----------')
            for k, v in sorted(args.items()):
                opt_file.write(f'%{str(k)}: {str(v)}')
            opt_file.write('-----------End----------')
        return self.opt