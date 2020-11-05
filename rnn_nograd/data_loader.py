import torch
import numpy as np
from .mnist import SequentialPixelMNIST, PermutedPixelMNIST
from .utils import set_allow_cuda


def npy_load_mnist(data_type, key):
    x = np.load(f'/home/carta/data/mnist/{data_type}_x_{key}.npy',)
    y = np.load(f'/home/carta/data/mnist/{data_type}_y_{key}.npy',)
    return x, y

def load_mnist(data):
    xs = []
    ys = []
    for x, y in data.iter():
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=1)
    ys = torch.cat(ys, dim=0)
    xs = xs.transpose(1, 0)
    return xs.numpy(), ys.numpy()


if __name__ == '__main__':
    data_dir = '/home/carta/data/mnist'
    set_allow_cuda(False)

    print("Preprocessing Sequential MNIST")
    for key in ['train', 'valid', 'test']:
        data = SequentialPixelMNIST(data_dir, set_key=key)
        x, y = load_mnist(data)
        np.save(data_dir + f'/seq_mnist_x_{key}.npy', x)
        np.save(data_dir + f'/seq_mnist_y_{key}.npy', y)

    print("Preprocessing Permuted MNIST")
    for key in ['train', 'valid', 'test']:
        data = PermutedPixelMNIST(data_dir, set_key=key, perm_file=data_dir+'/perm.pt')
        x, y = load_mnist(data)
        np.save(data_dir + f'/perseq_mnist_x_{key}.npy', x)
        np.save(data_dir + f'/perseq_mnist_y_{key}.npy', y)
