import torch
import training
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import netModel

training.parser.add_argument('--dataroot_lr_test', help='path to lr test dataset', default='./data/img_align_celeba55x45_test/')
training.parser.add_argument('--dataroot_hr_test', help='path to hr test dataset', default='./data/img_align_celeba180x220_test/')
training.parser.add_argument('--exp_name_reload', help='name of experiment to reload', default='test_1')
training.parser.add_argument('--which_epoch', help='name of experiment to reload', default='1')

opt = training.parser.parse_args()
opt.dataroot_lr = opt.dataroot_lr_test
opt.dataroot_hr = opt.dataroot_hr_test
opt.isTrain = False

# TODO check if downloading model
opt.exp_name = opt.exp_name_reload

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
    shuffle=True,
    num_workers=int(opt.workers))

model = netModel()
model.initialize(opt)

# Get one batch from each dataloader
data_lr = iter(dataloader_lr).next()
data_hr = iter(dataloader_hr).net()

model.set_input((data_hr, data_lr))
model.test()
visuals = model.get_current_visuals()

vutils.save_image(
    visuals['real_out'].data,
    '%s/real_samples.png' % (opt.outf + opt.exp_name),
    normalize=True)
vutils.save_image(
    visuals['fake_out'].data,
    '%s/fake_samples.png' % (opt.outf + opt.exp_name),
    normalize=True)
vutils.save_image(
    visuals['fake_in'].data,
    '%s/input_samples.png' % (opt.outf + opt.exp_name),
    normalize=True)
