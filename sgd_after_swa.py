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


parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--cuda_visible_devices', type=str, default='0', help='cuda_visible_devices (default: GPU0)')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--bnupdate', type=int, default=1, metavar='N', help='number of bnupdate (default: 1)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--lr_set', type=float, default=0.01, metavar='LR', help='set learning rate (default: 0.01)')
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
f_out = open (os.path.join(args.dir, 'output_record.txt'),'w')
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model, file = f_out)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path), file = f_out)
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

print('Preparing model', file = f_out)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()



print('SGD training', file = f_out)


criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_set,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume, file = f_out)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['swa_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

train_res_swa = utils.eval(loaders['train'], model, criterion)
test_res_swa = utils.eval(loaders['test'], model, criterion)
print(train_res_swa)
print(test_res_swa)
for epoch in range(start_epoch, start_epoch+50):
    time_ep = time.time()

    lr = args.lr_set
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)

    test_res = utils.eval(loaders['test'], model, criterion)


    if train_res['loss']<train_res_swa['loss'] and test_res['loss']>test_res_swa['loss']:
        print('find',file=f_out)
        print('find')
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )


    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table, file = f_out)
    print(table)


f_out.close()
