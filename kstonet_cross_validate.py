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
from process_data import preprocess_data

# Basic Setting
parser = argparse.ArgumentParser(description='Running K-StoNet')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--data_name', default='Boston', type=str, help='specify the name of the data')
parser.add_argument('--base_path', default='./result/', type=str,
                    help='base path for saving result')
parser.add_argument('--model_path', default='test_run/', type=str, help='folder name for saving model')
# parser.add_argument('--cross_validate', default=0, type=int, help='specify which fold of 5 fold cross validation')
parser.add_argument('--regression_flag', default=True, type=int,
                    help='true for regression and false for classification')
parser.add_argument('--normalize_y_flag', default=True, type=int, help='whether to normalize target value or not')
parser.add_argument('--confidence_interval_flag', default=False, type=int,
                    help='whether to store result to compute confidence interval')

# model
parser.add_argument('--layer', default=1, type=int, help='number of hidden layer')
parser.add_argument('--unit', default=[5], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[0.01], type=float, nargs='+',
                    help='variance of each layer for the hidden variable model')
# parser.add_argument('--C', default=5.0, type=float, help='C in SVR')

parser.add_argument('--C_list', default=[1.0, 2.0, 5.0, 10.0, 20.0], type=float, nargs='+', help='C in SVR for cross_validation')

parser.add_argument('--epsilon', default=0.01, type=float, help='epsilon in SVR')

# Training Setting
parser.add_argument('--nepoch', default=50, type=int, help='total number of training epochs')
parser.add_argument('--MH_step', default=25, type=int, help='SGLD step for imputation')
parser.add_argument('--lr', default=[0.0005], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--alpha', default=0.1, type=float, help='momentum parameter for HMC')
parser.add_argument('--temperature', default=[1], type=float, nargs='+', help='temperature parameter for HMC')
parser.add_argument('--lasso', default=[0.0001], type=float, nargs='+', help='lambda parameter for LASSO')

parser.add_argument('--p_gamma', default=None, type=float, help='inverse of gamma in the kernel')
parser.add_argument('--kernel', default='rbf', type=str,  help='kernel function of svr')


args = parser.parse_args()

def softplus(x):
    return (x.exp()+1).log()

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
        x = softplus(x)
        for i in range(self.num_hidden - 1):
            x = softplus(self.fc_list[i](x))
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_name = args.data_name

    num_hidden = args.layer
    hidden_dim = args.unit

    regression_flag = args.regression_flag
    normalize_y_flag = args.normalize_y_flag

    num_epochs = args.nepoch

    C_list = args.C_list



    for data_seed in range(0, 20):
        for cross_validate_index in range(10):
            x_data, y_data, x_test, y_test = preprocess_data(data_name, cross_validate_index, seed=data_seed)


            size_train = np.round(x_data.shape[0] * 0.8).astype(int)
            x_train = x_data[0:size_train, ]
            y_train = y_data[0:size_train,]
            x_val = x_data[size_train: , ]
            y_val = y_data[size_train: , ]


            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            ntrain = x_train.shape[0]
            nval = x_val.shape[0]
            ntest = x_test.shape[0]
            dim = x_train.shape[1]


            if regression_flag:
                if normalize_y_flag:
                    y_train_mean = y_train.mean()
                    y_train_std = y_train.std()
                    y_train = (y_train - y_train_mean) / y_train_std

            best_val_loss_list = np.zeros(len(C_list))
            for cross in range(len(C_list)):
                C = C_list[cross]
                print('C = ', C)
                sse = nn.MSELoss(reduction='sum')
                if regression_flag:
                    output_dim = 1
                    loss_func = nn.MSELoss()
                    loss_func_sum = nn.MSELoss(reduction='sum')
                    train_loss_path = np.zeros(num_epochs)
                    val_loss_path = np.zeros(num_epochs)
                    test_loss_path = np.zeros(num_epochs)
                else:
                    output_dim = int((y_test.max() + 1).item())
                    loss_func = nn.CrossEntropyLoss()
                    loss_func_sum = nn.CrossEntropyLoss(reduction='sum')
                    train_loss_path = np.zeros(num_epochs)
                    val_loss_path = np.zeros(num_epochs)
                    test_loss_path = np.zeros(num_epochs)
                    train_accuracy_path = np.zeros(num_epochs)
                    val_accuracy_path = np.zeros(num_epochs)
                    test_accuracy_path = np.zeros(num_epochs)
                time_used_path = np.zeros(num_epochs)

                net = Net(num_hidden, hidden_dim, output_dim)
                net.to(device)

                PATH = args.base_path + data_name + '/' + 'data_split_' + str(data_seed) + '/' + str(cross_validate_index) + '/' + 'C_' + str(
                    C) + args.model_path
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

                epsilon = args.epsilon


                temp_init = nn.Linear(dim, hidden_dim[0])
                temp_init.to(device)
                svr_output_init = temp_init(x_train)


                if args.p_gamma is None:
                    p_gamma = x_train.shape[1]
                else:
                    p_gamma = args.p_gamma

                kernel = args.kernel

                with torch.no_grad():
                    svr_list = []
                    for i in range(hidden_dim[0]):
                        temp = SVR(C=C, epsilon=epsilon, gamma=1.0/p_gamma, kernel = kernel)
                        svr_list.append(temp)

                svr_out_val = np.zeros([nval, hidden_dim[0]])

                svr_out_test = np.zeros([ntest, hidden_dim[0]])

                svr_out_train = svr_output_init.data.cpu().numpy()

                for epoch in range(num_epochs):

                    start_time = time.process_time()

                    hidden_list = []
                    momentum_list = []
                    with torch.no_grad():
                        hidden_list.append(torch.FloatTensor(svr_out_train).to(device))
                        momentum_list.append(torch.zeros_like(hidden_list[-1]))
                        for i in range(num_hidden - 1):
                            hidden_list.append(net.fc_list[i](softplus(hidden_list[-1])))
                            momentum_list.append(torch.zeros_like(hidden_list[-1]))

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
                                    net.fc_list[layer_index](softplus(hidden_list[layer_index])),
                                    y_train) / sigma_list[layer_index]
                            else:
                                hidden_likelihood = -sse(net.fc_list[layer_index](softplus(hidden_list[layer_index])),
                                                         hidden_list[layer_index + 1]) / sigma_list[layer_index]
                            if layer_index == 0:
                                hidden_likelihood = hidden_likelihood - C * torch.where(
                                    (hidden_list[layer_index] - forward_hidden).abs() - epsilon > 0,
                                    (hidden_list[layer_index] - forward_hidden).abs() - epsilon,
                                    torch.zeros_like(hidden_list[0])).sum()
                            else:
                                hidden_likelihood = hidden_likelihood - sse(
                                    net.fc_list[layer_index - 1](softplus(hidden_list[layer_index - 1])),
                                    hidden_list[layer_index]) / sigma_list[layer_index - 1]

                            hidden_likelihood.backward()
                            step_proposal_lr = proposal_lr[layer_index]
                            with torch.no_grad():
                                momentum_list[layer_index] = (1 - alpha) * momentum_list[
                                    layer_index] + step_proposal_lr / 2 * \
                                                             hidden_list[
                                                                 layer_index].grad + torch.FloatTensor(
                                    hidden_list[layer_index].shape).to(device).normal_().mul(
                                    np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
                                hidden_list[layer_index].data += momentum_list[layer_index]


                    with torch.no_grad():
                        for i in range(hidden_dim[0]):
                            svr_list[i].fit(x_train.cpu(), hidden_list[0][:, i].cpu().detach())

                    for layer_index in range(num_hidden):
                        if layer_index == num_hidden - 1:
                            if regression_flag:
                                clf = Lasso(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                                    softplus(hidden_list[layer_index]).cpu().detach(), y_train.cpu())

                                net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                                    net.fc_list[layer_index].weight.shape).to(device)
                                net.fc_list[layer_index].bias.data = torch.FloatTensor(
                                    np.array(clf.intercept_)).reshape(
                                    net.fc_list[layer_index].bias.shape).to(device)
                            else:
                                clf = LogisticRegression(penalty='l1', C=1 / lasso_lambda[layer_index], solver='saga',
                                                         max_iter=10000, multi_class='multinomial', n_jobs=5).fit(
                                    softplus(hidden_list[layer_index]).cpu().detach(), y_train.cpu())
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
                            clf = Lasso(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                                softplus(hidden_list[layer_index]).cpu().detach(),
                                hidden_list[layer_index + 1].cpu().detach())
                            net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                                net.fc_list[layer_index].weight.shape).to(device)
                            net.fc_list[layer_index].bias.data = torch.FloatTensor(clf.intercept_).reshape(
                                net.fc_list[layer_index].bias.shape).to(device)

                    end_time = time.process_time()
                    time_used_path[epoch] = end_time - start_time

                    with torch.no_grad():
                        if regression_flag:
                            print('epoch: ', epoch)
                            for i in range(hidden_dim[0]):
                                svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

                            output = net(torch.FloatTensor(svr_out_train).to(device))

                            train_loss = loss_func(output, y_train)
                            if normalize_y_flag:
                                train_loss = train_loss * y_train_std * y_train_std

                            train_loss_path[epoch] = train_loss
                            print("train loss: ", train_loss)

                            for i in range(hidden_dim[0]):
                                svr_out_test[:, i] = svr_list[i].predict(x_test.cpu())
                            output = net(torch.FloatTensor(svr_out_test).to(device))

                            if normalize_y_flag:
                                output = output * y_train_std + y_train_mean

                            test_loss = loss_func(output, y_test)
                            test_loss_path[epoch] = test_loss
                            print("test loss: ", test_loss)

                            for i in range(hidden_dim[0]):
                                svr_out_val[:, i] = svr_list[i].predict(x_val.cpu())

                            output = net(torch.FloatTensor(svr_out_val).to(device))
                            if normalize_y_flag:
                                output = output * y_train_std + y_train_mean
                            val_loss = loss_func(output, y_val)


                            val_loss_path[epoch] = val_loss
                            print("val loss: ", val_loss)

                        else:
                            print('epoch: ', epoch)
                            for i in range(hidden_dim[0]):
                                svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

                            output = net(torch.FloatTensor(svr_out_train).to(device))
                            train_loss = loss_func(output, y_train)
                            prediction = output.data.max(1)[1]
                            train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain

                            train_loss_path[epoch] = train_loss
                            train_accuracy_path[epoch] = train_accuracy
                            print("train loss: ", train_loss, 'train accuracy: ', train_accuracy)

                            for i in range(hidden_dim[0]):
                                svr_out_test[:, i] = svr_list[i].predict(x_test.cpu())

                            output = net(torch.FloatTensor(svr_out_test).to(device))
                            test_loss = loss_func(output, y_test)
                            prediction = output.data.max(1)[1]
                            test_accuracy = prediction.eq(y_test.data).sum().item() / ntest

                            test_loss_path[epoch] = test_loss
                            test_accuracy_path[epoch] = test_accuracy
                            print("test loss: ", test_loss, 'test accuracy: ', test_accuracy)

                            for i in range(hidden_dim[0]):
                                svr_out_val[:, i] = svr_list[i].predict(x_val.cpu())

                            output = net(torch.FloatTensor(svr_out_val).to(device))
                            val_loss = loss_func(output, y_val)
                            prediction = output.data.max(1)[1]
                            val_accuracy = prediction.eq(y_val.data).sum().item() / nval

                            val_loss_path[epoch] = val_loss
                            val_accuracy_path[epoch] = val_accuracy
                            print("val loss: ", val_loss, 'val accuracy: ', val_accuracy)


                    torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

                    for i in range(hidden_dim[0]):
                        filename = PATH + 'model_svr' + str(epoch) + '_' + str(i) + '.pt'
                        f = open(filename, 'wb')
                        pickle.dump(svr_list[i], f, protocol=4)
                        f.close()

                    if args.confidence_interval_flag:
                        filename = PATH + 'hidden_state' + str(epoch) + '.pt'
                        f = open(filename, 'wb')
                        pickle.dump(hidden_list, f, protocol=4)
                        f.close()

                    time_used_path[epoch] = end_time - start_time

                    if regression_flag:
                        filename = PATH + 'result.txt'
                        f = open(filename, 'wb')
                        pickle.dump([train_loss_path,val_loss_path, test_loss_path, time_used_path], f)
                        f.close()
                    else:
                        filename = PATH + 'result.txt'
                        f = open(filename, 'wb')
                        pickle.dump(
                            [train_loss_path, test_loss_path,val_loss_path, train_accuracy_path, val_accuracy_path, test_accuracy_path, time_used_path],
                            f)
                        f.close()
                    if args.confidence_interval_flag:
                        filename = PATH + 'data.txt'
                        f = open(filename, 'wb')
                        pickle.dump(
                            [x_train, x_test, x_val, y_val, y_train, y_test], f)
                        f.close()
                best_val_loss_list[cross] = val_loss_path.min()

            best_cross = np.argmin(best_val_loss_list)
            C = C_list[best_cross]

            print('best C = ', C)


            x_train = x_data
            y_train = y_data
            x_val = x_test
            y_val = y_test

            ntrain = x_train.shape[0]
            nval = x_val.shape[0]
            ntest = x_test.shape[0]
            dim = x_train.shape[1]


            sse = nn.MSELoss(reduction='sum')
            if regression_flag:
                output_dim = 1
                loss_func = nn.MSELoss()
                loss_func_sum = nn.MSELoss(reduction='sum')
                train_loss_path = np.zeros(num_epochs)
                val_loss_path = np.zeros(num_epochs)
                test_loss_path = np.zeros(num_epochs)
                if normalize_y_flag:
                    y_train_mean = y_train.mean()
                    y_train_std = y_train.std()
                    y_train = (y_train - y_train_mean) / y_train_std
            else:
                output_dim = int((y_test.max() + 1).item())
                loss_func = nn.CrossEntropyLoss()
                loss_func_sum = nn.CrossEntropyLoss(reduction='sum')
                train_loss_path = np.zeros(num_epochs)
                val_loss_path = np.zeros(num_epochs)
                test_loss_path = np.zeros(num_epochs)
                train_accuracy_path = np.zeros(num_epochs)
                val_accuracy_path = np.zeros(num_epochs)
                test_accuracy_path = np.zeros(num_epochs)
            time_used_path = np.zeros(num_epochs)

            net = Net(num_hidden, hidden_dim, output_dim)
            net.to(device)


            PATH = args.base_path + data_name + '/' + 'data_split_' + str(data_seed) + '/' + str(
                cross_validate_index) + '/' + 'best_C_' + str(
                C) + args.model_path
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

            epsilon = args.epsilon

            temp_init = nn.Linear(dim, hidden_dim[0])
            temp_init.to(device)
            svr_output_init = temp_init(x_train)


            if args.p_gamma is None:
                p_gamma = x_train.shape[1]
            else:
                p_gamma = args.p_gamma

            kernel = args.kernel

            with torch.no_grad():
                svr_list = []
                for i in range(hidden_dim[0]):

                    temp = SVR(C=C, epsilon=epsilon, gamma=1.0/p_gamma, kernel = kernel)
                    svr_list.append(temp)

            svr_out_val = np.zeros([nval, hidden_dim[0]])

            svr_out_test = np.zeros([ntest, hidden_dim[0]])

            svr_out_train = svr_output_init.data.cpu().numpy()

            for epoch in range(num_epochs):

                start_time = time.process_time()

                hidden_list = []
                momentum_list = []
                with torch.no_grad():
                    hidden_list.append(torch.FloatTensor(svr_out_train).to(device))
                    momentum_list.append(torch.zeros_like(hidden_list[-1]))
                    for i in range(num_hidden - 1):
                        hidden_list.append(net.fc_list[i](softplus(hidden_list[-1])))
                        momentum_list.append(torch.zeros_like(hidden_list[-1]))

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
                                net.fc_list[layer_index](softplus(hidden_list[layer_index])),
                                y_train) / sigma_list[layer_index]
                        else:
                            hidden_likelihood = -sse(net.fc_list[layer_index](softplus(hidden_list[layer_index])),
                                                     hidden_list[layer_index + 1]) / sigma_list[layer_index]
                        if layer_index == 0:
                            hidden_likelihood = hidden_likelihood - C * torch.where(
                                (hidden_list[layer_index] - forward_hidden).abs() - epsilon > 0,
                                (hidden_list[layer_index] - forward_hidden).abs() - epsilon,
                                torch.zeros_like(hidden_list[0])).sum()
                        else:
                            hidden_likelihood = hidden_likelihood - sse(
                                net.fc_list[layer_index - 1](softplus(hidden_list[layer_index - 1])),
                                hidden_list[layer_index]) / sigma_list[layer_index - 1]

                        hidden_likelihood.backward()
                        step_proposal_lr = proposal_lr[layer_index]
                        with torch.no_grad():
                            momentum_list[layer_index] = (1 - alpha) * momentum_list[
                                layer_index] + step_proposal_lr / 2 * \
                                                         hidden_list[
                                                             layer_index].grad + torch.FloatTensor(
                                hidden_list[layer_index].shape).to(device).normal_().mul(
                                np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
                            hidden_list[layer_index].data += momentum_list[layer_index]


                with torch.no_grad():
                    for i in range(hidden_dim[0]):
                        svr_list[i].fit(x_train.cpu(), hidden_list[0][:, i].cpu().detach())


                for layer_index in range(num_hidden):
                    if layer_index == num_hidden - 1:
                        if regression_flag:
                            clf = Lasso(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                                softplus(hidden_list[layer_index]).cpu().detach(), y_train.cpu())

                            net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                                net.fc_list[layer_index].weight.shape).to(device)
                            net.fc_list[layer_index].bias.data = torch.FloatTensor(
                                np.array(clf.intercept_)).reshape(
                                net.fc_list[layer_index].bias.shape).to(device)
                        else:
                            clf = LogisticRegression(penalty='l1', C=1 / lasso_lambda[layer_index], solver='saga',
                                                     max_iter=10000, multi_class='multinomial', n_jobs=5).fit(
                                softplus(hidden_list[layer_index]).cpu().detach(), y_train.cpu())
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
                        clf = Lasso(alpha=lasso_lambda[layer_index], max_iter=-1).fit(
                            softplus(hidden_list[layer_index]).cpu().detach(),
                            hidden_list[layer_index + 1].cpu().detach())
                        net.fc_list[layer_index].weight.data = torch.FloatTensor(clf.coef_).reshape(
                            net.fc_list[layer_index].weight.shape).to(device)
                        net.fc_list[layer_index].bias.data = torch.FloatTensor(clf.intercept_).reshape(
                            net.fc_list[layer_index].bias.shape).to(device)

                end_time = time.process_time()
                time_used_path[epoch] = end_time - start_time


                with torch.no_grad():
                    if regression_flag:
                        print('epoch: ', epoch)
                        for i in range(hidden_dim[0]):
                            svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

                        output = net(torch.FloatTensor(svr_out_train).to(device))


                        train_loss = loss_func(output, y_train)
                        if normalize_y_flag:
                            train_loss = train_loss * y_train_std * y_train_std

                        train_loss_path[epoch] = train_loss
                        print("train loss: ", train_loss)

                        for i in range(hidden_dim[0]):
                            svr_out_test[:, i] = svr_list[i].predict(x_test.cpu())

                        output = net(torch.FloatTensor(svr_out_test).to(device))

                        if normalize_y_flag:
                            output = output * y_train_std + y_train_mean

                        test_loss = loss_func(output, y_test)
                        test_loss_path[epoch] = test_loss
                        print("test loss: ", test_loss)

                        for i in range(hidden_dim[0]):
                            svr_out_val[:, i] = svr_list[i].predict(x_val.cpu())

                        output = net(torch.FloatTensor(svr_out_val).to(device))
                        if normalize_y_flag:
                            output = output * y_train_std + y_train_mean
                        val_loss = loss_func(output, y_val)

                        val_loss_path[epoch] = val_loss
                        print("val loss: ", val_loss)

                    else:
                        print('epoch: ', epoch)
                        for i in range(hidden_dim[0]):
                            svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

                        output = net(torch.FloatTensor(svr_out_train).to(device))
                        train_loss = loss_func(output, y_train)
                        prediction = output.data.max(1)[1]
                        train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain

                        train_loss_path[epoch] = train_loss
                        train_accuracy_path[epoch] = train_accuracy
                        print("train loss: ", train_loss, 'train accuracy: ', train_accuracy)

                        for i in range(hidden_dim[0]):
                            svr_out_test[:, i] = svr_list[i].predict(x_test.cpu())

                        output = net(torch.FloatTensor(svr_out_test).to(device))
                        test_loss = loss_func(output, y_test)
                        prediction = output.data.max(1)[1]
                        test_accuracy = prediction.eq(y_test.data).sum().item() / ntest

                        test_loss_path[epoch] = test_loss
                        test_accuracy_path[epoch] = test_accuracy
                        print("test loss: ", test_loss, 'test accuracy: ', test_accuracy)

                        for i in range(hidden_dim[0]):
                            svr_out_val[:, i] = svr_list[i].predict(x_val.cpu())

                        output = net(torch.FloatTensor(svr_out_val).to(device))
                        val_loss = loss_func(output, y_val)
                        prediction = output.data.max(1)[1]
                        val_accuracy = prediction.eq(y_val.data).sum().item() / nval

                        val_loss_path[epoch] = val_loss
                        val_accuracy_path[epoch] = val_accuracy
                        print("val loss: ", val_loss, 'val accuracy: ', val_accuracy)

                torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

                for i in range(hidden_dim[0]):
                    filename = PATH + 'model_svr' + str(epoch) + '_' + str(i) + '.pt'
                    f = open(filename, 'wb')
                    pickle.dump(svr_list[i], f, protocol=4)
                    f.close()

                if args.confidence_interval_flag:
                    filename = PATH + 'hidden_state' + str(epoch) + '.pt'
                    f = open(filename, 'wb')
                    pickle.dump(hidden_list, f, protocol=4)
                    f.close()

                time_used_path[epoch] = end_time - start_time

                if regression_flag:
                    filename = PATH + 'result.txt'
                    f = open(filename, 'wb')
                    pickle.dump([train_loss_path, val_loss_path, test_loss_path, time_used_path], f)
                    f.close()
                else:
                    filename = PATH + 'result.txt'
                    f = open(filename, 'wb')
                    pickle.dump(
                        [train_loss_path, test_loss_path, val_loss_path, train_accuracy_path, val_accuracy_path,
                         test_accuracy_path, time_used_path],
                        f)
                    f.close()
                if args.confidence_interval_flag:
                    filename = PATH + 'data.txt'
                    f = open(filename, 'wb')
                    pickle.dump(
                        [x_train, x_test, x_val, y_val, y_train, y_test], f)
                    f.close()


if __name__ == '__main__':
    main()






