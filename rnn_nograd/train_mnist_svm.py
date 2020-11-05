from sklearn.metrics import accuracy_score

import cannon
from cannon.utils import set_gpu, set_allow_cuda
from cannon.model_selection import ParamListTrainer
import shutil
from cannon.experiment import Experiment
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from cannon.laes import LinearAutoencoder
from sklearn.svm import SVC
from rnn_nograd.data_loader import npy_load_mnist
import os

set_gpu()


class LAES_SVM(BaseEstimator):

    def __init__(self, mem_size, C=1.0, gamma=0.1, nonlinear=False, pretrained_laes_file=None):
        super().__init__()
        self.is_fitted_ = False
        self.mem_size = mem_size
        self.C = C
        self.gamma = gamma
        self.nonlinear = nonlinear
        self.pretrained_laes_file = pretrained_laes_file

        self.laes = None
        if os.path.exists(pretrained_laes_file):
            self.laes = LinearAutoencoder(mem_size)
            self.laes.load(pretrained_laes_file)

    def fit(self, X, y):
        self.is_fitted_ = True

        if self.laes is None:
            self.laes = LinearAutoencoder(self.mem_size)
            self.laes.fit(X, svd_algo='cols', approx_k=110, verbose=True)
            if self.pretrained_laes_file is not None:
                self.laes.save(self.pretrained_laes_file)
        H = self.laes.encode(X)
        self.svm = self._train_svm(H, y)
        return self

    def _train_svm(self, H, y):
        if self.nonlinear:
            svm = SVC(C=self.C, gamma=self.gamma, kernel='rbf')
        else:
            svm = SVC(C=self.C, kernel='linear')
        return svm.fit(H, y)

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        H = self.laes.encode(X)
        y = self.svm.predict(H)
        return y

    def predict_all(self, X):
        check_is_fitted(self, 'is_fitted_')
        H = self.laes.encode(X)
        y = self.svm.predict(H)
        return y

    def __str__(self):
        return f"LAES_SVM({self.__dict__})"


def train_foo(log_dir, params):
    experiment_log = cannon.experiment.create_experiment_logger(log_dir)

    DEBUG = params['DEBUG']
    experiment_log.info("Loading Data.")
    if params['data_type'] == 'perseq_mnist':
        laes_file = './logs/perseq_mnist_laes_mem128.pkl'
        x_train, y_train = npy_load_mnist(params['data_type'], 'train')
        x_valid, y_valid = npy_load_mnist(params['data_type'], 'valid')
        x_test, y_test = npy_load_mnist(params['data_type'], 'test')
    elif params['data_type'] == 'seq_mnist':
        laes_file = './logs/seq_mnist_laes_mem128.pkl'
        x_train, y_train = npy_load_mnist(params['data_type'], 'train')
        x_valid, y_valid = npy_load_mnist(params['data_type'], 'valid')
        x_test, y_test = npy_load_mnist(params['data_type'], 'test')
    experiment_log.info("Data Loaded.")

    if DEBUG:
        log_dir = f'./logs/debug/'

    model = LAES_SVM(
        mem_size=params['mem_size'],
        C=params['C'],
        gamma=params['gamma'],
        nonlinear=params['nonlinear'],
        pretrained_laes_file=laes_file
    )
    experiment_log.info("Training Model.")
    model.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(x_train))
    experiment_log.info(f"TRAIN set performance: acc {train_acc}")
    valid_acc = accuracy_score(y_valid, model.predict(x_valid))
    experiment_log.info(f"VALID set performance: acc {valid_acc}")
    test_acc = accuracy_score(y_test, model.predict(x_test))
    experiment_log.info(f"TEST  set performance: acc {test_acc}")
    return {'tr_loss': -train_acc, 'tr_acc': train_acc, 'vl_loss': -valid_acc, 'vl_acc': valid_acc}


if __name__ == '__main__':
    set_allow_cuda(False)
    DEBUG = False
    model_type = 'laes_svm'
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
    }

    param_list = []
    for nonlinear in [True, False]:
        for C in [4 ** el for el in range(-4, 4)]:
            for gamma in [4 ** el for el in range(-4, 4)]:
                if nonlinear and gamma != 0:
                    continue
                pi = params.copy()
                pi['C'] = C
                pi['gamma'] = gamma
                pi['nonlinear'] = nonlinear
                param_list.append(pi)

    list_trainer = ParamListTrainer(log_dir, param_list, train_foo, resume_ok=True)
    list_trainer.run()

