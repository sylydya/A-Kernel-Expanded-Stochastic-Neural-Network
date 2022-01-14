import argparse
import os
import errno

import torch
import torch.nn as nn
import numpy as np

import torch.utils.data
import time

from thundersvm import SVR
# from joblib import Parallel, delayed


# from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from process_data import preprocess_data

# Basic Setting
parser = argparse.ArgumentParser(description='Running K-StoNet')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--data_name', default='CoverType', type=str, help='specify the name of the data')
parser.add_argument('--base_path', default='./result/', type=str,
                    help='base path for saving result')
parser.add_argument('--load_model_path',
                    default='test_run/', type=str,
                    help='folder name for loading model')
parser.add_argument('--model_path',
                    default='test_run/', type=str,
                    help='folder name for saving model')
parser.add_argument('--cross_validate', default=0, type=int, help='specify which fold of 5 fold cross validation')
parser.add_argument('--regression_flag', default=False, type=int,
                    help='true for regression and false for classification')

#model
parser.add_argument('--layer', default=1, type=int, help='number of hidden layer')
parser.add_argument('--unit', default=[50], type=int, nargs='+', help='number of hidden unit in each layer')

parser.add_argument('--C', default=10.0, type=float, help='C in SVR')
parser.add_argument('--epsilon', default=0.01, type=float, help='epsilon in SVR')

# Training Setting
parser.add_argument('--nepoch', default=10, type=int, help='total number of training epochs')

parser.add_argument('--load_epoch', default=-1, type=int, help='epoch of the loaded model')

parser.add_argument('--MH_step', default=25, type=int, help='SGLD step for imputation')
parser.add_argument('--lr', default=[0.00005], type=float, nargs='+', help='step size in imputation')

parser.add_argument('--sigma', default=[0.005], type=float, nargs='+',
                    help='variance of each layer for the hidden variable model')

parser.add_argument('--alpha', default=0.1, type=float, help='momentum parameter for HMC')
parser.add_argument('--temperature', default=[0.001], type=float, nargs='+', help='temperature parameter for HMC')
parser.add_argument('--lasso', default=[0.0001], type=float, nargs='+', help='lambda parameter for LASSO')

parser.add_argument('--kernel', default='rbf', type=str, help='kernel function of svr')
parser.add_argument('--p_gamma', default=None, type=float, help='inverse of gamma in the kernel')

parser.add_argument('--solve_index', default=0, type=int, help='index of svr')

args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.num_hidden = num_hidden
        self.fc_list = []
        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        x = torch.tanh(x)
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x


def svr_solver(svr_list, X, Y, i):
    svr_list[i].fit(X, Y[:, i])
    return svr_list[i]


def lasso_solver(weight, bias, solver, device, X, Y, i):
    clf = solver.fit(X, Y[:, i])
    dim = weight.shape[1]
    weight.data[i, :] = torch.FloatTensor(clf.coef_).reshape(dim).to(device)
    bias.data[i] = torch.FloatTensor(clf.intercept_).to(device)


def main():
    import pickle
    seed = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_name = args.data_name
    cross_validate_index = args.cross_validate

    num_hidden = args.layer
    hidden_dim = args.unit

    regression_flag = args.regression_flag
    num_epochs = args.nepoch

    # data_name = 'CoverType'
    x_train, y_train, x_test, y_test = preprocess_data(data_name, cross_validate_index, seed=seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    dim = x_train.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_test = x_test.to(device)
    y_test = y_test.to(device)


    sse = nn.MSELoss(reduction='sum')
    if regression_flag:
        output_dim = 1
        loss_func = nn.MSELoss()
        loss_func_sum = nn.MSELoss(reduction='sum')

    else:
        output_dim = int((y_test.max() + 1).item())
        loss_func = nn.CrossEntropyLoss()
        loss_func_sum = nn.CrossEntropyLoss(reduction='sum')


    net = Net(num_hidden, hidden_dim, output_dim)
    net.to(device)

    PATH = args.base_path + data_name + '/' + str(cross_validate_index) + '/' + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    proposal_lr = args.lr
    sigma_list = args.sigma
    temperature = args.temperature
    lasso_lambda = args.lasso

    if len(proposal_lr) == 1 and num_hidden > 1:
        temp_proposal_lr = proposal_lr[0]
        proposal_lr = []
        for i in range(num_hidden):
            proposal_lr.append(temp_proposal_lr)

    if len(sigma_list) == 1 and num_hidden > 1:
        temp_sigma_list = sigma_list[0]
        sigma_list = []
        for i in range(num_hidden):
            sigma_list.append(temp_sigma_list)

    if len(temperature) == 1 and num_hidden > 1:
        temp_temperature = temperature[0]
        temperature = []
        for i in range(num_hidden):
            temperature.append(temp_temperature)

    if len(lasso_lambda) == 1 and num_hidden > 1:
        temp_lasso_lambda = lasso_lambda[0]
        lasso_lambda = []
        for i in range(num_hidden):
            lasso_lambda.append(temp_lasso_lambda)

    C = args.C

    different_C = torch.ones(hidden_dim[0]).to(device) * C

    # for i in range(hidden_dim[0]):
    #     if i >= 0 and i < 30:
    #         C = 10
    #         different_C[i] = C
    #     if i >= 30 and i < 40:
    #         C = 10
    #         different_C[i] = C
    #     if i >= 40 and i < 45:
    #         C = 10
    #         different_C[i] = C
    #     if i >= 45 and i < 50:
    #         C = 10
    #         different_C[i] = C

    epsilon = args.epsilon

    temp_init = nn.Linear(dim, hidden_dim[0])
    temp_init.to(device)
    svr_output_init = temp_init(x_train)

    svr_list = []
    kernel = args.kernel

    load_epoch = args.load_epoch
    if load_epoch >= 0:
        load_PATH = args.base_path + data_name + '/' + str(cross_validate_index) + '/' + args.load_model_path

        while not os.path.exists(load_PATH + 'model' + str(load_epoch) + '.pt'):
            time.sleep(60)
        net.load_state_dict(torch.load(load_PATH + 'model' + str(load_epoch) + '.pt'))

        svr_out_train_path = load_PATH + 'svr_out_train_' + str(load_epoch) + '.pt'
        if os.path.exists(svr_out_train_path):
            filename = svr_out_train_path
            f = open(filename, 'rb')
            svr_out_train = pickle.load(f)
            f.close()
        else:
            for i in range(hidden_dim[0]):
                print('load model:', i)
                filename = load_PATH + 'model_svr' + str(load_epoch) + '_' + str(i) + '.pt'
                while not os.path.exists(filename):
                    time.sleep(60)
                f = open(filename, 'rb')
                temp = pickle.load(f)
                f.close()
                svr_list.append(temp)
            svr_out_train = np.zeros([ntrain, hidden_dim[0]])
            with torch.no_grad():
                print('epoch: ', load_epoch)
                for i in range(hidden_dim[0]):
                    svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())
                if not os.path.exists(svr_out_train_path):
                    filename = svr_out_train_path
                    f = open(filename, 'wb')
                    pickle.dump(svr_out_train, f)
                    f.close()

        with torch.no_grad():
            output = net(torch.FloatTensor(svr_out_train).to(device))
            train_loss = loss_func(output, y_train)
            prediction = output.data.max(1)[1]
            train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain

            print("train loss: ", train_loss, 'train accuracy: ', train_accuracy)

    else:
        svr_out_train = svr_output_init.data.cpu().numpy()


    for epoch in range(load_epoch + 1, load_epoch + 2):

        hidden_list = []
        momentum_list = []
        with torch.no_grad():
            hidden_list.append(torch.FloatTensor(svr_out_train).to(device))
            momentum_list.append(torch.zeros_like(hidden_list[-1]))
            for i in range(num_hidden - 1):
                hidden_list.append(net.fc_list[i](torch.tanh(hidden_list[-1])))
                momentum_list.append(torch.zeros_like(hidden_list[-1]))

        hidden_list_debug = []
        with torch.no_grad():
            for i in range(len(hidden_list)):
                hidden_list_debug.append(torch.clone(hidden_list[i]))

        for i in range(hidden_list.__len__()):
            hidden_list[i].requires_grad = True

        MH_step = args.MH_step
        alpha = args.alpha

        forward_hidden = torch.FloatTensor(svr_out_train).to(device)

        for repeat in range(MH_step):
            for layer_index in reversed(range(num_hidden)):
                if hidden_list[layer_index].grad is not None:
                    hidden_list[layer_index].grad.zero_()

                if layer_index == num_hidden - 1:
                    hidden_likelihood = -loss_func_sum(
                        net.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                        y_train) / sigma_list[layer_index]
                else:
                    hidden_likelihood = -sse(net.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                                             hidden_list[layer_index + 1]) / sigma_list[layer_index]
                if layer_index == 0:
                    hidden_likelihood = hidden_likelihood - (different_C * torch.where(
                        (hidden_list[layer_index] - forward_hidden).abs() - epsilon > 0,
                        (hidden_list[layer_index] - forward_hidden).abs() - epsilon,
                        torch.zeros_like(hidden_list[0]))).sum()
                else:
                    hidden_likelihood = hidden_likelihood - sse(
                        net.fc_list[layer_index - 1](torch.tanh(hidden_list[layer_index - 1])),
                        hidden_list[layer_index]) / sigma_list[layer_index - 1]

                hidden_likelihood.backward()
                step_proposal_lr = proposal_lr[layer_index]
                with torch.no_grad():
                    momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + step_proposal_lr / 2 * \
                                                 hidden_list[
                                                     layer_index].grad + torch.FloatTensor(
                        hidden_list[layer_index].shape).to(device).normal_().mul(
                        np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
                    hidden_list[layer_index].data += momentum_list[layer_index]

        a = 1
        with torch.no_grad():
            i = args.solve_index
            if i >= 0 and i < 30:
                p_gamma = 0.5
            if i >= 30 and i < 40:
                p_gamma = 1
            if i >= 40 and i < 45:
                p_gamma = 2
            if i >= 45 and i < 50:
                p_gamma = 5

            print('pgamma = ', p_gamma)
            temp_output = torch.clone(hidden_list[0][:, i]).cpu().detach()
            temp = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=1 / p_gamma)
            temp.fit(x_train.cpu(), temp_output)
            filename = PATH + 'model_svr' + str(epoch) + '_' + str(i) + '.pt'
            f = open(filename, 'wb')
            pickle.dump(temp, f, protocol=4)
            f.close()

        if args.solve_index == 0:
            for layer_index in range(num_hidden):
                if layer_index == num_hidden - 1:
                    if regression_flag:
                        clf = Ridge(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                            torch.tanh(hidden_list[layer_index]).cpu().detach(), y_train.cpu())

                        net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                            net.fc_list[layer_index].weight.shape).to(device)
                        net.fc_list[layer_index].bias.data = torch.FloatTensor(clf.intercept_).reshape(
                            net.fc_list[layer_index].bias.shape).to(device)
                    else:
                        clf = LogisticRegression(penalty='l2', C=1 / lasso_lambda[layer_index], solver='saga',
                                                 max_iter=10000, multi_class='multinomial', n_jobs=5).fit(
                            torch.tanh(hidden_list[layer_index]).cpu().detach(), y_train.cpu())

                        if output_dim == 2:
                            net.fc_list[layer_index].weight.data = torch.FloatTensor(
                                np.vstack([-clf.coef_, clf.coef_])).reshape(
                                net.fc_list[layer_index].weight.shape).to(device)
                            net.fc_list[layer_index].bias.data = torch.FloatTensor(
                                np.array([-clf.intercept_, clf.intercept_])).reshape(
                                net.fc_list[layer_index].bias.shape).to(device)
                        else:
                            net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                                net.fc_list[layer_index].weight.shape).to(device)
                            net.fc_list[layer_index].bias.data = torch.FloatTensor(clf.intercept_).reshape(
                                net.fc_list[layer_index].bias.shape).to(device)

                else:
                    clf = Ridge(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                        torch.tanh(hidden_list[layer_index]).cpu().detach(),
                        hidden_list[layer_index + 1].cpu().detach())

                    net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                        net.fc_list[layer_index].weight.shape).to(device)
                    net.fc_list[layer_index].bias.data = torch.FloatTensor(clf.intercept_).reshape(
                        net.fc_list[layer_index].bias.shape).to(device)

            torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')



if __name__ == '__main__':
    main()






