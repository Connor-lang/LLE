from turtle import forward
from unittest import skip
import numpy as np
import torch 
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
import util.util as util 
import itertools
from util.image_pool import ImagePool 
import sys 
import random

class EnlightenGANModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        batchSize = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(batchSize, opt.input_nc, size, size)
        self.input_B = self.Tensor(batchSize, opt.output_nc, size, size)
        self.input_img = self.Tensor(batchSize, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(batchSize, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.G = networks.UnetGenerator(opt.input_nc, opt.output_nc, self.gpu_ids)
        self.G.apply(lambda m: util.weights_init())

        self.D = networks.NLayerDiscriminator(opt.output_nc, n_layers=5)
        self.D.apply(lambda m: util.weights_init())

        self.patch_D = networks.NLayerDiscriminator(opt.output_nc, n_layers=4, padw=2)
        self.patch_D.apply(lambda m: util.weights_init())

        if not self.isTrain or opt.continue_train: 
            which_epoch = opt.which_epoch
            self.load_network(self.G, 'G', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool= ImagePool(opt.pool_size)
            
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_patch_D = torch.optim.Adam(self.patch_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    def set_input(self, input):
        self.input_A = input['A']
        self.input_A_gray = input['A_gray']
        self.input_B = input['B']
        self.input_img = input['input_img']
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        self.fake_B, self.latent_real_A = self.G.forward(self.real_A, self.real_A_gray)
        self.real_B = Variable(self.input_B, volatile=True)
    
    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
    
    def get_image_paths(self):
        return self.image_paths
    
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)

        self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        self.fake_B, self.latent_real_A = self.G(self.real_img, self.input_A_gray)
        
        if self.opt.patchD:
            h = self.input_A.size(2)
            w = self.input_A.size(3)
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            
            self.fake_patch = self.fake_B[:, :, h_offset:h_offset + self.opt.patchSize, 
                                            w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:, :, h_offset:h_offset + self.opt.patchSize, 
                                            w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:, :, h_offset:h_offset + self.opt.patchSize, 
                                            w_offset:w_offset + self.opt.patchSize]

        if self.opt.n_patch > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            
            for i in range(self.opt.n_patch):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

    def backward_D_basic(self, netD, real, fake, use_ragan):
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) + 
                        self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.D, self.real_B, fake_B, True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.patch_D, self.real_patch, self.fake_patch, False)
            if self.opt.n_patch > 0:
                for i in range(self.opt.n_patch):
                    loss_D_P += self.backward_D_basic(self.patch_D, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P / float(self.opt.n_patch + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.patch_D, self.real_patch, self.fake_patch, True)
            if self.opt.n_patch > 0:
                for i in range(self.opt.n_patch):
                    loss_D_P += self.backward_D_basic(self.patch_D, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P / float(self.opt.n_patch + 1)
            else:
                self.loss_D_P = loss_D_P
        self.loss_D_P.backward()
    
    def backward_G(self, epoch):
        pred_fake = self.D.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            pred_real = self.D.forward(self.real_B)
            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) + 
                                self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.patch_D.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.patch_D.forward(self.real_patch)
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.n_patch > 0:
            for i in range(self.opt.n_patch):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
            self.loss_G_A += loss_G_A / float(self.opt.n_patch + 1) * 2
        else:
            self.loss_G_A += loss_G_A * 2
        
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, 
                                    self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_VGG:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.n_patch > 0:
                    for i in range(self.opt.n_patch):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.n_patch + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b * vgg_w
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        #G
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        #D
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D.step()
        else:
            self.optimizer_patch_D.zero_grad()
            self.backward_D_P()
            self.backward_D_A.step()
            self.backward_D_P.step()
    
    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.data
        D_P = self.loss_D_P.data[0] if self.opt.patchD else 0
        G = self.loss_G_A.data[0]
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.data[0] / self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('D_A', D_A), ('G', G), ('vgg', vgg), ('D_P', D_P)])
        else:
            return OrderedDict([('D_A', D_A), ('G', G), ('D_P', D_P)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        latent_real_A = util.latent2im(self.latent_real_A.data)
        fake_patch = util.tensor2im(self.fake_patch.data)
        real_patch = util.tensor2im(self.real_patch.data)
        self_attention = util.atten2im(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), ('real_B', real_B), 
                                ('real_patch', real_patch), ('fake_patch', fake_patch), ('self_attention', self_attention)])
        
    def save(self, label):
        self.save_network(self.G, 'G', label, self.gpu_ids)
        self.save_network(self.D, 'D', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.patch_D, 'D_P', label, self.gpu_ids)
    
    def update_learning_rate(self):
        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_patch_D.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('Update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr