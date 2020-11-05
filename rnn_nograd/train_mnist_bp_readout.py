from sklearn.metrics import accuracy_score
import cannon
from cannon.tasks.mnist import PermutedPixelMNIST, SequentialPixelMNIST
from cannon.torch_trainer import SequentialTaskTrainer
from cannon.utils import set_gpu, set_allow_cuda, cuda_move
from cannon.model_selection import ParamListTrainer
import shutil
from cannon.experiment import Experiment
from sklearn.utils.validation import check_is_fitted
from cannon.laes import LinearAutoencoder
from sklearn.svm import SVC
from rnn_nograd.data_loader import npy_load_mnist
from torch import nn
import os
import torch

set_gpu()


class TorchLAESFixed:
    def __init__(self, A, B, bias):
        self.hidden_size = B.shape[0]
        self.A = cuda_move(torch.tensor(A).float())
        self.B = cuda_move(torch.tensor(B).float())
        self.bias = cuda_move(torch.tensor(bias).float())

    def __call__(self, x):
        h0 = cuda_move(torch.zeros(x.shape[1], self.hidden_size))
        for ti in range(x.shape[0]):
            h0 = (x[ti] - self.bias) @ self.A.T + h0 @ self.B.T
        return h0


class LAESDeepReadout(nn.Module):
    def __init__(self, mem_size, num_layers, num_classes, pretrained_laes_file):
        super().__init__()
        self.is_fitted_ = False
        self.mem_size = mem_size

        self.layers = []
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(mem_size, mem_size))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(mem_size, num_classes))

        self.ro = nn.Sequential(*self.layers)

        self.np_laes = None
        self.np_laes = LinearAutoencoder(mem_size)
        self.np_laes.load(pretrained_laes_file)
        self.laes = TorchLAESFixed(self.np_laes.A, self.np_laes.B, self.np_laes.mean)

    def forward(self, x):
        x = self.laes(x).detach()
        return self.ro(x)


def train_foo(log_dir, params):
    experiment_log = cannon.experiment.create_experiment_logger(log_dir)

    DEBUG = params['DEBUG']
    experiment_log.info("Loading Data.")
    if params['data_type'] == 'perseq_mnist':
        laes_file = './logs/perseq_mnist_laes_mem128.pkl'
        train_data = PermutedPixelMNIST(data_dir, 'train', DEBUG, perm_file=data_dir+'/perm.pt')
        val_data = PermutedPixelMNIST(data_dir, 'valid', DEBUG, perm_file=data_dir+'/perm.pt')
        test_data = PermutedPixelMNIST(data_dir, 'test', DEBUG, perm_file=data_dir+'/perm.pt')
    elif params['data_type'] == 'seq_mnist':
        laes_file = './logs/seq_mnist_laes_mem128.pkl'
        train_data = SequentialPixelMNIST(data_dir, 'train', DEBUG)
        val_data = SequentialPixelMNIST(data_dir, 'valid', DEBUG)
        test_data = SequentialPixelMNIST(data_dir, 'test', DEBUG)
    experiment_log.info("Data Loaded.")

    if DEBUG:
        log_dir = f'./logs/debug/'
        params['n_epochs'] = 5

    model = LAESDeepReadout(
        mem_size=params['mem_size'],
        num_layers=params['num_layers'],
        num_classes=10,
        pretrained_laes_file=laes_file
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    trainer = SequentialTaskTrainer(
        cuda_move(model),
        n_epochs=params['n_epochs'],
        optimizer=optimizer,
        log_dir=log_dir
    )
    trainer.append_hyperparam_dict(params)
    trainer.fit(train_data, val_data)
    te, ta = trainer.compute_metrics(test_data)
    trainer.logger.info(f"TEST set performance: loss {te}, acc {ta}")
    return trainer.best_result


if __name__ == '__main__':
    set_allow_cuda(False)
    DEBUG = False
    model_type = 'laes_tbp_readout'
    data_type = 'perseq_mnist'
    data_dir = '/home/carta/data/mnist'
    project_name = f'{data_type}_{model_type}'
    log_dir = f'./logs/{data_type}/{model_type}/'

    if DEBUG:
        log_dir = './logs/debug/'
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

    shutil.copytree('./rnn_nograd', log_dir + 'src')
    params = {
        'DEBUG': DEBUG,
        'project_name': project_name,
        'model_type': model_type,
        'data_type': data_type,
        'log_dir': log_dir,
        'in_size': 1,
        'mem_size': 128,
        'learning_rate': 1.e-3,
        'weight_decay': 1.e-5,
        'n_epochs': 100
    }

    param_list = []
    for num_layers in [1, 2, 3]:
        pi = params.copy()
        pi['num_layers'] = num_layers
        param_list.append(pi)

    list_trainer = ParamListTrainer(log_dir, param_list, train_foo, resume_ok=True)
    list_trainer.run()
