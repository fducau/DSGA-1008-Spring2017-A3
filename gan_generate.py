import torch
import training
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import netModel
import urllib2
import os
import zipfile

training.parser.add_argument('--dataroot_hr_test', help='path to hr test dataset', default='./data/img_align_celeba180x220_test/')
training.parser.add_argument('--dataroot_lr_test', help='path to lr test dataset', default='./data/img_align_celeba55x45_test')
training.parser.add_argument('--exp_name_reload', help='name of experiment to reload', default='7_lambda0-1_nlayers3_kw2_accuracylimits')
training.parser.add_argument('--which_epoch', help='name of experiment to reload', default='7')

opt = training.parser.parse_args()
opt.dataroot_hr = opt.dataroot_hr_test
opt.dataroot_lr = opt.dataroot_lr_test
opt.isTrain = False


# Retrieve data
print('Retrieving dataset from internet')

try:
    os.makedirs('data')
except OSError:
    pass

# Download zip file
response = urllib2.urlopen('https://s3.amazonaws.com/emrbucket-fnd212/DL_HW/SISR/SISR_celeba_testdata.zip')
zipfile = response.read()
f = open('./data/SISR_celeba_testdata_S3.zip', 'wb')
f.write(zipfile)
f.close()

# Unzip data
zip_ref = zipfile.ZipFile('./data/SISR_celeba_testdata_S3.zip', 'r')
zip_ref.extractall('./data/')
zip_ref.close()

print('Retrieving trained Generator')
response = urllib2.urlopen('https://s3.amazonaws.com/emrbucket-fnd212/DL_HW/SISR/7_lambda0-1_nlayers3_kw2_accuracylimits/netG_epoch_9.pth')
saved_model = response.read()
f = open(opt.outf + opt.exp_name + '/netG_epoch_9.pth', 'wb')
f.write(saved_model)
f.close()


# TODO check if downloading model
opt.exp_name = opt.exp_name_reload
opt.batchSize=50

print('test_time:', opt)

dataset_hr = dset.ImageFolder(
    root=opt.dataroot_hr,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

assert dataset_hr

dataset_size = len(dataset_hr)
dataloader_hr = torch.utils.data.DataLoader(
    dataset_hr,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

dataset_lr = dset.ImageFolder(
    root=opt.dataroot_lr,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

assert dataset_lr

dataloader_lr = torch.utils.data.DataLoader(
    dataset_lr,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

# Get one batch from each dataloader
data_hr = iter(dataloader_hr).next()
data_lr = iter(dataloader_lr).next()

model = netModel()
model.initialize(opt)

model.set_input((data_hr, data_lr))
model.test()
visuals = model.get_current_visuals()

vutils.save_image(
    data_hr[0],
    '%s/real_test_samples.png' % (opt.outf + opt.exp_name),
    nrow=5, normalize=True)
vutils.save_image(
    visuals['fake_out'].data,
    '%s/fake_test_samples_epoch%s.png' % (opt.outf + opt.exp_name, opt.which_epoch),
    nrow=5, normalize=True)
vutils.save_image(
    visuals['fake_in'].data,
    '%s/input_test_samples.png' % (opt.outf + opt.exp_name),
    nrow=5, normalize=True)
