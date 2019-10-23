"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

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


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

# torch.manual_seed(1)    # reproducible

parser = argparse.ArgumentParser(description='LR 2 params')
# parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--num_epoches', type=int, default=30, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--dir', type=str, default='logistic_regression_2params', required=False, help='training directory (default: None)')
# parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name (default: MNIST)')
# parser.add_argument('--data_path', type=str, default='data', required=False, metavar='PATH',
#                     help='path to datasets location (default: None)')
# parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
#                     help='checkpoint to resume training from (default: None)')
# parser.add_argument('--load_swa_model', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--distances_scale', type=float, default=0.5, metavar='N', help='explore distance scale (default: 1)')

args = parser.parse_args()

batch_size = 10

def find_asym(seed0):
    os.makedirs(args.dir, exist_ok=True)
    torch.manual_seed(seed0)  #2 3  =====================================================================================
    torch.cuda.manual_seed(7)
    # make fake data
    n_data = torch.ones(batch_size, 1)
    x0 = torch.normal(n_data, 1)      # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(batch_size)               # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-n_data, 1)     # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(batch_size)                # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.FloatTensor)    # shape (200,) LongTensor = 64-bit integer
    print('x', x)
    y = y.view(-1, 1)
    print('y',y)



    class Net(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(Net, self).__init__()
            self.linear = torch.nn.Linear(n_in, n_out)   # hidden layer
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            y_pred = self.sigmoid(self.linear(x))
            return y_pred

    net = Net(1, 1)     # define the network
    net_swa = Net(1, 1)     # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.BCELoss(size_average=True)

    swa_start_epoch = 1
    swa_n=0
    for t in range(args.num_epoches):
        out = net(x)                 # input x and predict based on x
        loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
        # print(out)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        print(t, loss)


        moving_average(net_swa, net, 1.0 / (swa_n + 1))
        swa_n += 1

        save_checkpoint(
            args.dir,
            t,
            state_dict=net.state_dict(),
            swa_state_dict=net_swa.state_dict(),
            # swa_n=swa_n,
            # optimizer=optimizer.state_dict()
        )

    #explore
    vec_sgd = parameters_to_vector(net.parameters())
    vec_swa = parameters_to_vector(net_swa.parameters())
    print(vec_swa.size())
    vec_rand = torch.rand(vec_swa.shape)
    vec_rand = vec_rand / torch.norm(vec_rand)

    print(vec_sgd)
    print(vec_swa)

    loss_record = np.zeros(31)
    # vec_rand = vec_swa - vec_sgd
    # distances_scale = torch.norm(vec_rand)/5

    for distance in range(-15, 15 + 1):
        print(distance)
        vec_temp = vec_swa + distance * vec_rand * args.distances_scale
        vector_to_parameters(vec_temp, net.parameters())

        net.eval()
        out = net(x)  # input x and predict based on x
        loss_record[distance+15] = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

    np.savetxt(os.path.join(args.dir, 'loss_record.txt'), loss_record)


    #draw figure with format?
    sgd_train_loss_results = np.loadtxt(os.path.join(args.dir, 'loss_record.txt'))
    distances = 15
    distances_scale = args.distances_scale
    plt.rcParams['figure.figsize'] = (7.0, 4.0)
    plt.subplots_adjust(bottom=.12, top=.99, left=.1, right=.99)
    plt.plot(np.arange(-distances*distances_scale, distances*distances_scale + distances_scale, distances_scale), sgd_train_loss_results, label='Training loss', color='dodgerblue')
    plt.scatter(0, sgd_train_loss_results[distances], marker='o',s=70,c='orange',label='SGD solution')
    plt.legend(fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.xlabel('A random direction generated from (0,1)-uniform distribution',fontsize=13)
    plt.savefig(os.path.join(args.dir, 'logistic_regression_asym'+str(seed0)+'.png'))
    plt.savefig(os.path.join(args.dir, 'logistic_regression_asym'+str(seed0)+'.pdf'))
    plt.close()

for i in range(50):  #50
    find_asym(i)

