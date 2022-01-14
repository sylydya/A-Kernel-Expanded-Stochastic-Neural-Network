
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.utils.data

import pickle

import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

def compute_distance(X, Y):
    n_1, n_2 = X.size(0), Y.size(0)

    norms_1 = torch.sum(X ** 2, dim=1, keepdim=True)
    norms_2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2))
    distances_squared = norms - 2 * X @ Y.T

    return torch.abs(distances_squared)

def RBF_kernel(x, y , gamma):
    distance = compute_distance(x, y)
    kernel = torch.exp(-gamma * distance)
    return kernel


def RBF_kernel_data(x_train, x_val=None, x_test=None, gamma=None):
    if gamma is None:
        gamma = 1 / (x_train.shape[1] * x_train.var())

    distances_train = compute_distance(x_train, x_train)
    print('haha')
    kernel_train = torch.exp(-gamma * distances_train)

    if x_val is not None:
        distances_val = compute_distance(x_val, x_train)
        kernel_val = torch.exp(-gamma * distances_val)
    else:
        kernel_val = None
    print('haha2')

    if x_test is not None:
        distances_test = compute_distance(x_test, x_train)
        kernel_test = torch.exp(-gamma * distances_test)
    else:
        kernel_test = None
    return kernel_train, kernel_val, kernel_test


def dtanh(x):
    return 1-torch.tanh(x).pow(2)


TotalP = 5
sigma = 1
NTest = 500
x_test = np.matrix(np.zeros([NTest, TotalP]))
y_test = np.matrix(np.zeros([NTest, 1]))

for i in range(NTest):
    ee = np.sqrt(sigma) * np.random.normal(0, 1)
    for j in range(TotalP):
        x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)
        # x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / 2.0
    x0 = x_test[i, 0]
    x1 = x_test[i, 1]
    x2 = x_test[i, 2]
    x3 = x_test[i, 3]
    x4 = x_test[i, 4]

    y_test[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    x_test[i, 0] = x_test[i, 0] + np.random.normal(0, 0.5)
    x_test[i, 1] = x_test[i, 1] + np.random.normal(0, 0.5)
    x_test[i, 2] = x_test[i, 2] + np.random.normal(0, 0.5)
    x_test[i, 3] = x_test[i, 3] + np.random.normal(0, 0.5)
    x_test[i, 4] = x_test[i, 4] + np.random.normal(0, 0.5)
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

ntest = 500
num_seed = 100
num_epoch = 25
dim = 5
hidden_dim = 5
net = Net(dim)
count_list = np.zeros([ntest, num_epoch])

lower_bound_list = np.zeros([ntest, num_seed, num_epoch])
upper_bound_list = np.zeros([ntest, num_seed, num_epoch])
net.to(device)

for test_index in range(ntest):
    print('test index = ', test_index)
    z = x_test[test_index,].unsqueeze(0)
    y_z = y_test[test_index]

    count = 0
    for seed in range(1, 101):
        for epoch in range(49, 50):
            # epoch = 49
            base_path = './result/measurement_error/0/'
            model_path = 'seed' + str(seed) + '/'

            PATH = base_path + model_path

            svr_list = []
            for i in range(hidden_dim):
                filename = PATH + 'model_svr' + str(epoch) + '_' + str(i) + '.pt'
                f = open(filename, 'rb')
                temp = pickle.load(f)
                f.close()
                svr_list.append(temp)

            filename = PATH + 'hidden_state' + str(epoch) + '.pt'
            f = open(filename, 'rb')
            hidden_list = pickle.load(f)
            f.close()

            filename = PATH + 'data.txt'
            f = open(filename, 'rb')
            [x_train, x_val, x_test, y_train, y_val, y_test] = pickle.load(f)
            f.close()

            filename = PATH + 'result.txt'
            f = open(filename, 'rb')
            [train_loss_path, test_loss_path, time_used_path] = pickle.load(f)
            f.close()

            net.load_state_dict(torch.load(PATH + 'model' + str(epoch) + '.pt'))

            epsilon = 0.05
            C = 1
            num_hidden = 5
            hidden_sigma_1 = torch.zeros([num_hidden, num_hidden])
            hidden_mu_1 = torch.zeros([num_hidden])
            for hidden_index in range(5):
                margin_vector = x_train[svr_list[hidden_index].support_,][
                    np.where(np.abs(svr_list[hidden_index].dual_coef_) < 1)[1],]

                K_XM_XZ = RBF_kernel(margin_vector, z, gamma=1 / (x_train.shape[1] * x_train.var()))
                K_XM_XM = RBF_kernel(margin_vector, margin_vector, gamma=1 / (x_train.shape[1] * x_train.var()))
                K_XZ_XZ = RBF_kernel(z, z, gamma=1 / (x_train.shape[1] * x_train.var()))

                temp_sigma = K_XZ_XZ - K_XM_XZ.transpose(0, 1).matmul(K_XM_XM.inverse()).matmul(K_XM_XZ)

                hidden_sigma_1[hidden_index, hidden_index] = temp_sigma
                hidden_mu_1[hidden_index] = svr_list[hidden_index].predict(z)[0]

            hidden_state = hidden_list[0].cpu().data

            tanh_impute = torch.tanh(hidden_state)

            impute_inverse = tanh_impute.transpose(0, 1).matmul(tanh_impute).inverse()

            dtanh_z = torch.diag(dtanh(hidden_mu_1))
            tanh_z = torch.tanh(hidden_mu_1.unsqueeze(0))

            term1 = impute_inverse.matmul(dtanh_z).matmul(hidden_sigma_1).matmul(dtanh_z)

            term2 = tanh_z.matmul(impute_inverse).matmul(tanh_z.transpose(0, 1))

            linear_regression_sigma = train_loss_path[epoch]

            w_1 = net.fc2.weight.data
            term3 = w_1.matmul(dtanh_z).matmul(hidden_sigma_1).matmul(dtanh_z).matmul(w_1.transpose(0, 1))

            hidden_sigma_2 = (term1.trace() + term2) * linear_regression_sigma + term3

            hidden_mu_2 = net(hidden_mu_1).data

            z_score = 1.96

            lower_bound = hidden_mu_2 - (hidden_sigma_2 + train_loss_path[epoch]).sqrt() * z_score
            upper_bound = hidden_mu_2 + (hidden_sigma_2 + train_loss_path[epoch]).sqrt() * z_score


            if lower_bound < y_z and upper_bound > y_z:
                count_list[test_index, epoch - 25] = count_list[test_index, epoch - 25] + 1

            lower_bound_list[test_index, seed - 1, epoch - 25] = lower_bound
            upper_bound_list[test_index, seed - 1, epoch - 25] = upper_bound



index = np.random.choice(500, 20, replace=False)

temp_x_axis = np.arange(20)


temp_x = x_test[:,0].numpy()
temp_y = y_test.numpy().reshape(temp_x.shape)

index = temp_x.argsort()
subindex = np.arange(0, 500, 25)


temp_lower = temp_y - lower_bound_list[:, 0, -1]
temp_upper = upper_bound_list[:, 0, -1] - temp_y
temp_predict = (lower_bound_list[:, 0, -1] + upper_bound_list[:, 0, -1])/2

temp_error = np.stack([temp_lower, temp_upper])



plt.errorbar(temp_x_axis, temp_y[index[subindex]],yerr=temp_error[:,index[subindex]], fmt='*', capsize=5, ecolor='r')
plt.plot(temp_x_axis, temp_y[index[subindex]], 'b*', label = 'True Value')
plt.plot(temp_x_axis, temp_predict[index[subindex]], 'ro', label = 'Predicted Value')

plt.xlabel('index')
plt.ylabel('Y')
plt.legend(loc="upper right")
plt.title('Confidence Interval')

plt.xticks(np.arange(0, 20, 1.0))
