from .laes import LinearAutoencoder
from .mnist import SequentialPixelMNIST, PermutedPixelMNIST
from .utils import set_allow_cuda
import numpy as np
import os


def load_data(data):
    X = []
    y = []
    for xi, yi in data.iter():
        assert len(xi.shape) == 3
        assert xi.shape[1] == 1
        X.append(xi[:, 0, :].cpu().numpy())
        y.append(yi.cpu())
        if len(X) == n:
            break
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    return X, y


if __name__ == '__main__':
    set_allow_cuda(False)
    DEBUG = False
    data_dir = '/home/carta/data/mnist'
    data_type = 'perseq_mnist'
    memory_size = 128
    n = 60000
    log_file = f'./logs/{data_type}_laes_mem{memory_size}.pkl'

    print("Loading Data.")
    if data_type == 'seq_mnist':
        train_data = SequentialPixelMNIST(data_dir, 'train', debug=DEBUG, batch_size=1)
        val_data = SequentialPixelMNIST(data_dir, 'valid', debug=DEBUG, batch_size=1)
    elif data_type == 'perseq_mnist':
        train_data = PermutedPixelMNIST(data_dir, 'train', debug=DEBUG, batch_size=1, perm_file=data_dir+'/perm.pt')
        val_data = PermutedPixelMNIST(data_dir, 'valid', debug=DEBUG, batch_size=1, perm_file=data_dir+'/perm.pt')

    X, y = load_data(train_data)
    if not os.path.exists(log_file):
        print("Training Autoencoder.")
        la = LinearAutoencoder(p=memory_size)
        la.fit(X, svd_algo='fb_pca', approx_k=128)
        la.save(log_file)
    else:
        print("loading LA from existing save.")
        la = LinearAutoencoder(p=memory_size)
        la.load(log_file)

    print("Training linear regressor.")
    h_train = la.encode(X)
    y_onehot = np.eye(10)[y[:, 0]]
    W, _, _, _ = np.linalg.lstsq(h_train, y_onehot)
    y_pred = h_train @ W
    y_pred = np.argmax(y_pred, axis=1)
    acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]
    print(f"Train set ACC: {acc}")

    X_val, y_val = load_data(val_data)
    h_val = la.encode(X_val)

    y_pred = h_val @ W
    y_pred = np.argmax(y_pred, axis=1)
    acc = (y_pred.reshape(-1) == y_val.reshape(-1)).sum() / y_val.shape[0]
    print(f"VAL set ACC: {acc}")
