from cannon.regularizers import OrthogonalPenalty
from .mnist import SequentialPixelMNIST, PermutedPixelMNIST
from .utils import cuda_move, set_gpu, set_allow_cuda
from rnn_nograd.rnn_modules import *
from cannon.torch_trainer import SequentialTaskTrainer
from cannon.model_selection import ParamListTrainer
from cannon.callbacks import TrainingCallback, OrthogonalInit
import os
import shutil
import numpy as np
from cannon.laes.svd_la import LinearAutoencoder
set_gpu()


class LMNPretrainingRNN(TrainingCallback):
    def __init__(self, train_data, val_data, la_file):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.la_file = la_file
        self.la = LinearAutoencoder(128)
        self.la.load(la_file)

    def _encode_data(self, data):
        X = []
        y = []
        for xi, yi in data.iter():
            for bi in range(xi.shape[1]):
                X.append(xi[:, bi, :].cpu().numpy())
                y.append(yi[bi].unsqueeze(0).cpu().numpy())
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        return X, y, self.la.encode(X)

    def before_training(self, model_trainer):
        model = model_trainer.model.rnn.layer

        model_trainer.logger.info("Training linear regressor.")
        X, y, h_train = self._encode_data(self.train_data)
        y_onehot = np.eye(10)[y[:, 0]]
        W, _, _, _ = np.linalg.lstsq(h_train, y_onehot)

        y_pred = h_train @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_tr_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        X, y, h_val = self._encode_data(self.val_data)
        y_pred = h_val @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_vl_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        model_trainer.logger.info("Initializing matrices.")
        model.Wxh.data = torch.tensor(self.la.A).float()
        model.bh.data = torch.zeros_like(model.bh.data)
        model.Wmh.data = torch.zeros_like(model.Wmh.data)
        model.Whm.data = torch.eye(model.memory_size).float()
        model.Wmm.data = torch.tensor(self.la.B).float()
        model.bm.data = torch.tensor(-(self.la.A @ self.la.mean).reshape(-1)).float()
        model_trainer.model.Wo.data = torch.tensor(W.T).float()
        model_trainer.model.bo.data = torch.zeros_like(model_trainer.model.bo)
        cuda_move(model_trainer.model)

        tr_err, tr_acc = model_trainer.compute_metrics(self.train_data)
        vl_err, vl_acc = model_trainer.compute_metrics(self.val_data)
        model_trainer.logger.info("**************************************************************")
        model_trainer.logger.info(f"* Loaded checkpoint from: {self.la_file}")
        model_trainer.logger.info(f"* Linear RNN: TRAIN acc {la_tr_acc:.3f}")
        model_trainer.logger.info(f"* Linear RNN: VALID acc {la_vl_acc:.3f}")
        model_trainer.logger.info(f"* After model init: TRAIN loss {tr_err:.3f}, acc {tr_acc:.3f}")
        model_trainer.logger.info(f"* After model init: VALID loss {vl_err:.3f}, acc {vl_acc:.3f}")
        model_trainer.logger.info("**************************************************************")

    def __str__(self):
        return self.__class__.__name__ + '(TrainingCallback)'


def train_foo(log_dir, params):
    DEBUG = params['DEBUG']
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

    if DEBUG:
        log_dir = f'./logs/debug/'
        params['n_epochs'] = 5

    cbs, regs = [], []
    if model_type == 'lstm':
        rnn = LSTMLayer(
            in_size=params['in_size'],
            hidden_size=params['hidden_size']
        )
        with torch.no_grad():
            mask = torch.zeros_like(rnn.layer.bias_hh)
            mask[params['hidden_size']: 2*params['hidden_size']] = 1  # ingate, forgetgate, cellgate, outgate
            rnn.layer.bias_hh.masked_fill_(mask == 1, params['forget_bias'])
        assert rnn.layer.bias_hh.is_leaf
    elif model_type == 'lstm_detach':
        rnn = LSTMDetachLayer(
            in_size=params['in_size'],
            hidden_size=params['hidden_size'],
            p_detach=params['p_detach']
        )
        with torch.no_grad():
            mask = torch.zeros_like(rnn.layer.bias_hh)
            mask[params['hidden_size']: 2 * params['hidden_size']] = 1  # ingate, forgetgate, cellgate, outgate
            rnn.layer.bias_hh.masked_fill_(mask == 1, params['forget_bias'])
        assert rnn.layer.bias_hh.is_leaf
    elif model_type == 'lmn_detach':
        rnn = LMNDetachLayer(
            in_size=params['in_size'],
            hidden_size=params['hidden_size'],
            memory_size=params['hidden_size'],
            p_detach=params['p_detach']
        )
    elif model_type == 'lmn_ortho_detach':
        rnn = LMNDetachLayer(
            in_size=params['in_size'],
            hidden_size=params['hidden_size'],
            memory_size=params['hidden_size'],
            p_detach=params['p_detach']
        )
        cbs.append(OrthogonalInit([rnn.layer.Wmm]))
        regs.append(OrthogonalPenalty(params['ortho_penalty'], [rnn.layer.Wmm]))
    elif model_type == 'lmn_laes_detach':
        rnn = LMNDetachLayer(
            in_size=params['in_size'],
            hidden_size=params['hidden_size'],
            memory_size=params['hidden_size'],
            p_detach=params['p_detach']
        )
        cbs.append(LMNPretrainingRNN(train_data, val_data, laes_file))
        regs.append(OrthogonalPenalty(params['ortho_penalty'], [rnn.layer.Wmm]))

    model = cuda_move(SequenceClassifier(
        rnn_cell=rnn,
        hidden_size=params['hidden_size'],
        output_size=params['output_size']
    ))

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_decay'])
    trainer = SequentialTaskTrainer(
        cuda_move(model),
        n_epochs=params['n_epochs'],
        optimizer=optimizer,
        regularizers=regs,
        log_dir=log_dir,
        callbacks=cbs,
        grad_clip=params['grad_clip']
    )
    trainer.append_hyperparam_dict(params)
    trainer.fit(train_data, val_data)
    te, ta = trainer.compute_metrics(test_data)
    trainer.logger.info(f"TEST set performance: loss {te}, acc {ta}")
    return trainer.best_result


if __name__ == '__main__':
    set_allow_cuda(True)
    DEBUG = False
    model_type = 'lmn_laes_detach'
    data_type = 'seq_mnist'
    data_dir = '/home/carta/data/mnist'
    project_name = f'{data_type}_{model_type}'
    log_dir = f'./logs/{data_type}/{model_type}_v3/'

    if DEBUG:
        log_dir = './logs/debug/'
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

    shutil.copytree('./tbp', log_dir + 'src')
    params = {
        'DEBUG': DEBUG,
        'project_name': project_name,
        'model_type': model_type,
        'data_type': data_type,
        'log_dir': log_dir,
        'in_size': 1,
        'hidden_size': 128,
        'output_size': 10,
        'out_hidden': False,
        'learning_rate': 1.e-3,
        'l2_decay': 1.e-5,
        'n_epochs': 100,
        'grad_clip': 10,
        'p_detach': 0.0,
        'forget_bias': 1.0,
        'ortho_penalty': 0.0
    }

    param_list = []
    for grad_clip in [10]:  # [1, 10, 10 ** 10]:
        for lr in [1.e-4, 1.e-3]:
            for ortho_p in [0.001, 0.0001, 0.01, 0.1]:
                for p_detach in [0.0, .1, .25, .5, 1.0]:
                    pi = params.copy()
                    pi['learning_rate'] = lr
                    pi['grad_clip'] = grad_clip
                    pi['p_detach'] = p_detach
                    pi['ortho_penalty'] = ortho_p
                    param_list.append(pi)

    list_trainer = ParamListTrainer(log_dir, param_list, train_foo, resume_ok=True)
    list_trainer.run()

