import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
from torch.autograd import Variable
import networks
import util
from base_model import BaseModel




class netModel(BaseModel):
    def name(self):
        return 'netModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # self.isTrain = opt.isTrain
        # define tensors
        self.input_real = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.imageSize, opt.imageSize)
        self.input_fake = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.imageSize, opt.imageSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_real = input[0][0]
        input_fake = input[1][0]

        self.input_real.resize_(input_real.size()).copy_(input_real)
        self.input_fake.resize_(input_fake.size()).copy_(input_fake)

    def forward(self):
        self.fake_in = Variable(self.input_fake)
        self.fake_out = self.netG.forward(self.fake_in)
        self.real_out = Variable(self.input_real)

    # no backprop gradients
    def test(self):
        self.fake_in = Variable(self.input_fake, volatile=True)
        self.fake_out = self.netG.forward(self.fake_in)
        self.real_out = Variable(self.input_real, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        self.pred_fake = self.netD.forward(self.fake_out.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_fake.backward()
        # Real
        self.pred_real = self.netD.forward(self.real_out)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real.backward()
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake = self.netD.forward(self.fake_out)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_in, self.fake_out) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        # st()
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                ('G_L1', self.loss_G_L1.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
        ])

    def get_current_visuals(self):
        fake_in = util.tensor2im(self.fake_in.data)
        fake_out = util.tensor2im(self.fake_out.data)
        rereal_out = util.tensor2im(self.real_out.data)
        return OrderedDict([('fake_in', fake_in), ('fake_out', fake_out), ('real_out', real_out)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
