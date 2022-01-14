import argparse
import os
import errno

import torch
import torch.nn as nn
import numpy as np

import torch.utils.data
import time

from process_data import preprocess_data

# Basic Setting
parser = argparse.ArgumentParser(description='KNN ')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--data_name', default = 'Boston', type = str, help='specify the name of the data')
parser.add_argument('--base_path', default='./result/', type=str, help='base path for saving result')
parser.add_argument('--model_path', default='knn_test_run/', type=str, help='folder name for saving model')
parser.add_argument('--cross_validate', default=0, type = int, help='specify which fold of 5 fold cross validation')
parser.add_argument('--regression_flag', default=True, type=int, help='true for regression and false for classification')
parser.add_argument('--normalize_y_flag', default=True, type=int, help='whether to normalize target value or not')

# model
parser.add_argument('--layer', default=1, type=int, help='number of hidden layer')
parser.add_argument('--unit', default=[5], type=int, nargs='+', help='number of hidden unit in each layer')


# Training Setting
parser.add_argument('--nepoch', default = 40, type = int, help = 'total number of training epochs')
parser.add_argument('--lr', default = 0.0001, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--weight_decay', default = 0, type = float, help = 'weight decay in SGD')
parser.add_argument('--batch_train', default = 1, type = int, help = 'batch size for training')
parser.add_argument('--lasso', default=0, type=float, help='lambda parameter for LASSO')
parser.add_argument('--n_repeat', default=20, type=int, help='number of repeat')

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):
        super(Net, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x



def compute_distance(X, Y):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    X : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    Y : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| X[i, :] - Y[j, :] ||_2^2``."""
    n_1, n_2 = X.size(0), Y.size(0)

    # Compute (a - b)^2 = a^2 + b^2 -2ab
    norms_1 = torch.sum(X ** 2, dim=1, keepdim=True)
    norms_2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2))
    distances_squared = norms - 2 * X @ Y.T

    # Take the absolute value due to numerical imprecision
    return torch.abs(distances_squared)

def Gaussian_kernel(x, y, sigma = 1):
    return (x-y).pow(2).sum().mul(-(0.5/sigma)).exp()

def Gaussian_kernel_data(x_train, x_val = None, x_test = None, sigma = 1):
    distances_train = compute_distance(x_train, x_train)
    kernel_train = torch.exp( - distances_train / (2 * sigma))

    if x_val is not None:
        distances_val = compute_distance(x_val, x_train)
        kernel_val = torch.exp(- distances_val / (2 * sigma))
    else:
        kernel_val = None

    if x_test is not None:
        distances_test = compute_distance(x_test, x_train)
        kernel_test = torch.exp(- distances_test / (2 * sigma))
    else:
        kernel_test = None
    return kernel_train, kernel_val, kernel_test


def main():
    import pickle
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_name = args.data_name

    num_hidden = args.layer
    hidden_dim = args.unit

    regression_flag = args.regression_flag
    normalize_y_flag = args.normalize_y_flag

    num_epochs = args.nepoch

    for data_seed in range(args.n_repeat):
        for cross_validate_index in range(10):
            x_train_orig, y_train, x_test_orig, y_test = preprocess_data(data_name, cross_validate_index, seed=data_seed)
            dim = x_train_orig.shape[1]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            x_train, x_val, x_test = Gaussian_kernel_data(x_train_orig, None, x_test_orig,
                                                          sigma=0.5 * dim * x_train_orig.var())

            ntrain = x_train.shape[0]
            ntest = x_test.shape[0]
            dim = x_train.shape[1]

            x_train = (x_train - x_train.mean(0)) / x_train.std(0)
            x_test = (x_test - x_test.mean(0)) / x_test.std(0)

            if regression_flag:
                output_dim = 1
                loss_func = nn.MSELoss()
                train_loss_path = np.zeros(num_epochs)
                test_loss_path = np.zeros(num_epochs)
                if normalize_y_flag:
                    y_train_mean = y_train.mean()
                    y_train_std = y_train.std()
                    y_train = (y_train - y_train_mean) / y_train_std

            else:
                output_dim = int((y_test.max() + 1).item())
                loss_func = nn.CrossEntropyLoss()
                train_loss_path = np.zeros(num_epochs)
                test_loss_path = np.zeros(num_epochs)
                train_accuracy_path = np.zeros(num_epochs)
                test_accuracy_path = np.zeros(num_epochs)
            time_used_path = np.zeros(num_epochs)

            net = Net(num_hidden, hidden_dim, dim, output_dim)
            net.to(device)

            PATH = args.base_path + data_name + '/' + 'data_split_' + str(data_seed) + '/' + str(
                cross_validate_index) + '/' + 'knn/' + args.model_path

            if not os.path.isdir(PATH):
                try:
                    os.makedirs(PATH)
                except OSError as exc:  # Python >2.5
                    if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                        pass
                    else:
                        raise

            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            torch.manual_seed(args.seed)

            index = np.arange(ntrain)
            subn = args.batch_train

            lasso_lambda = args.lasso

            for epoch in range(num_epochs):
                start_time = time.process_time()
                np.random.shuffle(index)
                for iter in range(ntrain // subn):
                    subsample = index[(iter * subn):((iter + 1) * subn)]
                    optimizer.zero_grad()
                    loss = loss_func(net(x_train[subsample,]), y_train[subsample,])
                    loss.backward()

                    for para in net.parameters():
                        loss += para.abs().sum().mul(lasso_lambda)

                    optimizer.step()

                with torch.no_grad():
                    if regression_flag:
                        print('epoch: ', epoch)
                        output = net(x_train)
                        train_loss = loss_func(output, y_train)
                        if normalize_y_flag:
                            train_loss = train_loss * y_train_std * y_train_std
                        train_loss_path[epoch] = train_loss
                        print("train loss: ", train_loss)

                        output = net(x_test)
                        if normalize_y_flag:
                            output = output * y_train_std + y_train_mean
                        test_loss = loss_func(output, y_test)
                        test_loss_path[epoch] = test_loss
                        print("test loss: ", test_loss)

                    else:
                        print('epoch: ', epoch)
                        output = net(x_train)
                        train_loss = loss_func(output, y_train)
                        prediction = output.data.max(1)[1]
                        train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain
                        train_loss_path[epoch] = train_loss
                        train_accuracy_path[epoch] = train_accuracy
                        print("train loss: ", train_loss, 'train accuracy: ', train_accuracy)

                        output = net(x_test)
                        test_loss = loss_func(output, y_test)
                        prediction = output.data.max(1)[1]
                        test_accuracy = prediction.eq(y_test.data).sum().item() / ntest
                        test_loss_path[epoch] = test_loss
                        test_accuracy_path[epoch] = test_accuracy
                        print("test loss: ", test_loss, 'test accuracy: ', test_accuracy)

                torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

                end_time = time.process_time()

                time_used_path[epoch] = end_time - start_time

            if regression_flag:
                filename = PATH + 'result.txt'
                f = open(filename, 'wb')
                pickle.dump([train_loss_path, test_loss_path, time_used_path], f)
                f.close()
            else:
                filename = PATH + 'result.txt'
                f = open(filename, 'wb')
                pickle.dump([train_loss_path, test_loss_path, train_accuracy_path, test_accuracy_path, time_used_path],
                            f)
                f.close()




if __name__ == '__main__':
    main()






