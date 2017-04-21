'''
Base code from https://github.com/pytorch/examples/tree/master/dcgan
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import cycle
from itertools import izip
import time
from model import netModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot_real', help='path to dataset', default='./data/imagenet/')
parser.add_argument('--dataroot_fake', help='path to dataset', default='./data/fake/')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--L1lambda', type=float, default=10., help='Weight for L1 loss in generator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./runs/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
parser.add_argument('--norm', type=str, default='batch', help='batch normalization or instance normalization')


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# folder dataset
dataset_real = dset.ImageFolder(root=opt.dataroot_real,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

assert dataset_real
dataset_size = len(dataset_real)
dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))

# folder dataset
dataset_fake = dset.ImageFolder(root=opt.dataroot_fake,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

assert dataset_fake
dataloader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))


model = netModel()
model.initialize(opt)
print("model was created")
# Add visualizer?

total_steps = 0
for epoch in range(opt.niter):
    epoch_start_time = time.time()
    for i, (data, data_fake) in enumerate(izip(dataloader_real, cycle(dataloader_fake)), 0):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        model.set_input((data, data_fake))
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            visuals = model.get_current_visuals()

            vutils.save_image(visuals['real_out'],
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)
            vutils.save_image(visuals['fake_out'],
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)
            vutils.save_image(visuals['fake_in'],
                              '%s/input_samples.png' % opt.outf,
                              normalize=True)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            print('G_GAN: {}, G_L1: {}, D_real: {}, D_fake: {}'.format(
                  errors['G_GAN'], errors['G_L1'], errors['D_real'],
                  errors['D_fake']))


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        input_fake_cpu, _ = data_fake

        batch_size = real_cpu.size(0)

        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        input_fake.data.resize_(input_fake_cpu.size()).copy_(input_fake_cpu)

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG_autoencoder(input_fake)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        D_G_z1 = output.data.mean()

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG_autoencoder.zero_grad()
        #fake = netG_autoencoder(input_fake)

        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)# + img_similarity_cirterion(fake, input_fake)
        errG.backward(retain_variables=True)
        errG_similarity = img_similarity_cirterion(fake, input_fake) * opt.L1lambda
        errG_similarity.backward()
        D_G_z2 = output.data.mean()
        optimizerG_autoencoder.step()


        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f L1: %.4f'
              % (epoch, opt.niter, i, len(dataloader_real),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, errG_similarity.data[0]))
        if i % 10000 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG_autoencoder(fixed_fake_imgs)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
            vutils.save_image(fixed_fake_imgs.data,
                    '%s/input_samples.png' % opt.outf,
                    normalize=True)

    if epoch % 10 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))