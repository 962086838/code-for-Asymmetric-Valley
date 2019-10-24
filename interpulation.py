import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import models
import utils
import tabulate
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl

def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--cuda_visible_devices', type=str, default='0', help='cuda_visible_devices (default: GPU0)')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')

parser.add_argument('--distances', type=int, default=20, metavar='N', help='explore radius (default: 20)')
parser.add_argument('--division_part', type=int, default=40, metavar='N', help='division_part(default: 20)')
parser.add_argument('--distances_scale', type=float, default=1.0, metavar='N', help='explore scale (default: 1)')

parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--model1_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--model2_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
os.system('cp -r ./'+sys.argv[0]+' ./'+args.dir+'/')
f_out = open (os.path.join(args.dir, 'output_record.txt'),'w')
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())

def target_transform(target):
    # print('target', target)
    return int(target)
if args.dataset == 'SVHN':
    train_set = ds(path, split='train', download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                target_transform=target_transform,)
    test_set = ds(path, split='test', download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                target_transform=target_transform,)
else:
    train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
    test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}
if args.dataset == 'SVHN':
    num_classes = 10
else:
    num_classes = 10#max(train_set.train_labels) + 1

print('Preparing model')
model_1 = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model_2 = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model_temp = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model_direction = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

#model_temp = model
model_1.cuda()
model_2.cuda()
model_temp.cuda()
model_direction.cuda()




def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = F.cross_entropy
# optimizer = torch.optim.SGD(
#     model_temp.parameters(),
#     lr=args.lr_init,
#     momentum=args.momentum,
#     weight_decay=args.wd
# )



start_epoch = 0
if args.model1_resume is not None:
    print('Resume training from %s' % args.model1_resume)
    checkpoint = torch.load(args.model1_resume)
    start_epoch = checkpoint['epoch']
    model_1.load_state_dict(checkpoint['state_dict'])
    utils.bn_update(loaders['train'], model_1)
    print(utils.eval(loaders['train'], model_1, criterion))
vec_1 = parameters_to_vector(model_1.parameters())

if args.model2_resume is not None:
    print('Resume training from %s' % args.model2_resume)
    checkpoint = torch.load(args.model2_resume)
    start_epoch = checkpoint['epoch']
    model_2.load_state_dict(checkpoint['swa_state_dict'])
    model_temp.load_state_dict(checkpoint['state_dict'])
    utils.bn_update(loaders['train'], model_2)
    print(utils.eval(loaders['train'], model_2, criterion))
vec_2 = parameters_to_vector(model_2.parameters())

vec_inter = vec_1 - vec_2
# vec_inter_norm = torch.norm(vec_inter)
print(torch.norm(vec_inter), file=f_out)
vec_inter = vec_inter / args.division_part
f_out.flush()
#generate direction weight vector

dis_counter = 0
result_shape = args.distances * 2 + args.division_part + 1

train_loss_results_bnupdate = np.zeros(result_shape)
test_loss_results_bnupdate = np.zeros(result_shape)
train_acc_results_bnupdate = np.zeros(result_shape)
test_acc_results_bnupdate = np.zeros(result_shape)

for i in range(0, int(result_shape), 1):
    print(i)
    print(i, file=f_out)
    vec_temp = vec_2 + (i - args.distances) * vec_inter
    vector_to_parameters(vec_temp, model_temp.parameters())
    utils.bn_update(loaders['train'], model_temp)

    train_temp = utils.eval(loaders['train'], model_temp, criterion)
    test_temp = utils.eval(loaders['test'], model_temp, criterion)
    print(train_temp)
    print(train_temp, file=f_out)
    print(test_temp)
    print(test_temp, file=f_out)

    train_loss_results_bnupdate[dis_counter] = train_temp['loss']
    train_acc_results_bnupdate[dis_counter] = train_temp['accuracy']
    test_loss_results_bnupdate[dis_counter] = test_temp['loss']
    test_acc_results_bnupdate[dis_counter] = test_temp['accuracy']

    np.savetxt(os.path.join(args.dir, "train_loss_results.txt"), train_loss_results_bnupdate)
    np.savetxt(os.path.join(args.dir, "test_loss_results.txt"), test_loss_results_bnupdate)
    np.savetxt(os.path.join(args.dir, "train_acc_results.txt"), train_acc_results_bnupdate)
    np.savetxt(os.path.join(args.dir, "test_acc_results.txt"), test_acc_results_bnupdate)
    dis_counter += 1
    #print("test", test_temp)
    #print("sgd exploring on train and test set %d"%(dis_counter))
    f_out.flush()

plt.cla()
plt.plot(train_loss_results_bnupdate)
plt.savefig(os.path.join(args.dir, 'train_loss_results.png'))
plt.cla()
plt.plot(test_loss_results_bnupdate)
plt.savefig(os.path.join(args.dir, 'test_loss_results.png'))
plt.cla()
plt.plot(train_acc_results_bnupdate)
plt.savefig(os.path.join(args.dir, 'train_acc_results.png'))
plt.cla()
plt.plot(test_acc_results_bnupdate)
plt.savefig(os.path.join(args.dir, 'test_acc_results.png'))

f_out.close()
