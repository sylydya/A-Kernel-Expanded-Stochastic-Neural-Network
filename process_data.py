import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVR


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


def preprocess_data(data_name, cross_validate_index, seed = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if data_name == 'global_optimal':
        # device = torch.device("cpu")
        TotalP = 5
        NTrain = 5000
        n_unit = 5
        second_weight = np.mat(np.random.normal(0,1,[n_unit, 1]))
        second_bias = np.random.normal(0,1,1)

        x_train = np.mat(np.zeros([NTrain, TotalP]))
        sigma = 1.0
        for i in range(NTrain):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)

        C = 5
        epsilon = 0.01
        p_gamma = 1 / (x_train.shape[1] * x_train.var())

        x_train_hidden = np.mat(np.random.normal(0, 1, [NTrain, n_unit]))

        svr_input_train = RBF_kernel(torch.FloatTensor(x_train), torch.FloatTensor(x_train), p_gamma)
        svr_input_train = svr_input_train.numpy()

        dual_coef = np.mat(np.zeros([NTrain, n_unit]))
        dual_bias = np.zeros([n_unit])

        for i in range(n_unit):
            temp_svr = SVR(C=C, epsilon=epsilon)
            temp_svr.fit(x_train, x_train_hidden[:, i])

            dual_coef[temp_svr.support_, i] = temp_svr.dual_coef_
            dual_bias[i] = temp_svr.intercept_

        x_train_forward_hidden = np.matmul(svr_input_train, dual_coef) + dual_bias

        y_train = np.matmul(x_train_forward_hidden, second_weight) + second_bias + np.random.normal(0, 1, [NTrain, 1])

        NTest = 5000
        x_test = np.mat(np.zeros([NTest, TotalP]))
        for i in range(NTest):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)

        svr_input_test = RBF_kernel(torch.FloatTensor(x_test), torch.FloatTensor(x_train), p_gamma)

        svr_input_test = svr_input_test.numpy()

        x_test_forward_hidden = np.matmul(svr_input_test, dual_coef) + dual_bias

        y_test = np.matmul(x_test_forward_hidden, second_weight) + second_bias + np.random.normal(0, 1, [NTest, 1])

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)

        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)


    if data_name == 'measurement_error':
        TotalP = 5
        NTrain = 500
        x_train = np.matrix(np.zeros([NTrain, TotalP]))
        y_train = np.matrix(np.zeros([NTrain, 1]))

        sigma = 1.0
        for i in range(NTrain):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)
            x0 = x_train[i, 0]
            x1 = x_train[i, 1]
            x2 = x_train[i, 2]
            x3 = x_train[i, 3]
            x4 = x_train[i, 4]

            y_train[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

            x_train[i, 0] = x_train[i, 0] + np.random.normal(0, 0.5)
            x_train[i, 1] = x_train[i, 1] + np.random.normal(0, 0.5)
            x_train[i, 2] = x_train[i, 2] + np.random.normal(0, 0.5)
            x_train[i, 3] = x_train[i, 3] + np.random.normal(0, 0.5)
            x_train[i, 4] = x_train[i, 4] + np.random.normal(0, 0.5)

        NTest = 500
        x_test = np.matrix(np.zeros([NTest, TotalP]))
        y_test = np.matrix(np.zeros([NTest, 1]))

        for i in range(NTest):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)
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

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)

        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    if data_name == 'full_row_rank':
        TotalP = 1000
        a = 1
        b = 1
        W1 = np.matrix(np.random.choice([-2, -1, 1, 2], size=TotalP * 5, replace=True).reshape([TotalP, 5]))
        W2 = np.matrix(np.random.choice([-2, -1, 1, 2], size=5 * 5, replace=True).reshape([5, 5]))
        W3 = np.matrix(np.random.choice([-2, -1, 1, 2], size=5 * 1, replace=True).reshape([5, 1]))
        NTrain = 1000
        x_train = np.matrix(np.zeros([NTrain, TotalP]))
        y_train = np.matrix(np.zeros([NTrain, 1]))
        sigma = 1.0
        for i in range(NTrain):
            if i % 1000 == 0:
                print("x_train generate = ", i)
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)

        temp = np.tanh(x_train * W1)
        temp = np.tanh(temp * W2)
        y_train = temp * W3 + np.random.normal(0, 1, size=y_train.shape)

        NTest = 1000
        x_test = np.matrix(np.zeros([NTest, TotalP]))
        y_test = np.matrix(np.zeros([NTest, 1]))

        sigma = 1.0
        for i in range(NTest):
            if i % 1000 == 0:
                print("x_test generate = ", i)
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)

        temp = np.tanh(x_test * W1)
        temp = np.tanh(temp * W2)
        y_test = temp * W3 + np.random.normal(0, 1, size=y_test.shape)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)

        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)


    elif data_name == 'parkinson':
        temp = pd.read_table('./data/prakinson_telemonitoring/parkinsons_updrs.data', sep=',')

        temp = np.mat(temp)
        x_data = temp[:, 6:]
        y_data = temp[:, 5]

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])
        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)

        y_train = (y_train - y_train_mean) / y_train_std

        y_test = (y_test - y_train_mean) / y_train_std

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    elif data_name == 'qsar':

        temp = pd.read_csv('./data/qsar/qsar_androgen_receptor.csv', sep=';', header=None)
        temp = np.mat(temp)
        x_data = temp[:, 0:-1].astype('float64')
        y_data = temp[:, -1]

        y_data = (y_data == 'positive')

        y_data = np.array(y_data.astype('int')).reshape(y_data.shape[0])

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)


        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'MNIST':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        x_train = train_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).reshape(
            [train_set.data.shape[0], -1])
        x_test = test_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).reshape(
            [test_set.data.shape[0], -1])
        y_train = train_set.targets
        y_test = test_set.targets

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    elif data_name == "CoverType":

        device = torch.device("cpu")
        df = pd.read_csv('./data/CoverType/covtype.data', sep=',', header=None)

        y = df[54]
        X = df.drop(54, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y, train_size=0.5)


        x_train = np.array(X_train).astype('float64')
        x_test = np.array(X_test).astype('float64')
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train[:, 0:10] = (x_train[:, 0:10] - np.full(x_train[:, 0:10].shape, x_train_mean[0:10])) / np.full(
            x_train[:, 0:10].shape,
            x_train_std[0:10])

        x_test[:, 0:10] = (x_test[:, 0:10] - np.full(x_test[:, 0:10].shape, x_train_mean[0:10])) / np.full(
            x_test[:, 0:10].shape, x_train_std[0:10])


        x_train = torch.FloatTensor(x_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        y_train = y_train - 1
        y_test = torch.LongTensor(y_test).to(device)
        y_test = y_test - 1
    elif data_name == "Boston":
        temp = np.loadtxt('./data/Boston/housing.data')
        x_data = temp[:, 0:-1]
        y_data = temp[:, -1].reshape([temp.shape[0], 1])

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    elif data_name == 'Concrete':
        temp = pd.read_excel('data/Concrete/Concrete_Data.xls')

        temp = np.mat(temp)
        x_data = temp[:, :8]
        y_data = temp[:, 8]

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    elif data_name == 'Energy':
        temp = pd.read_excel('data/Energy/ENB2012_data.xlsx')

        temp = np.mat(temp)
        x_data = temp[:768, :8]
        y_data = temp[:768, 9]


        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'Wine':
        temp = pd.read_csv('data/Wine/winequality-red.csv', sep=';')

        temp = np.mat(temp)
        x_data = temp[:, 0:11]
        y_data = temp[:, 11]


        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'Yacht':

        temp = np.loadtxt('data/Yacht/yacht_hydrodynamics.data')
        x_data = temp[:, 0:6]
        y_data = temp[:, 6].reshape([temp.shape[0], 1])

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'kin8nm':
        temp = pd.read_csv('data/kin8nm/dataset_2175_kin8nm.csv')

        temp = np.mat(temp)
        x_data = temp[:, 0:8]
        y_data = temp[:, 8]


        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'Naval':

        temp = np.loadtxt('data/Naval/data.txt')
        x_data = temp[:, 0:16]
        y_data = temp[:, 16].reshape([temp.shape[0], 1])


        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'CCPP':
        temp = pd.read_excel('data/CCPP/Folds5x2_pp.xlsx')

        temp = np.mat(temp)
        x_data = temp[:, :4]
        y_data = temp[:, 4]

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    elif data_name == 'Protein':
        temp = pd.read_csv('data/Protein/CASP.csv')

        temp = np.matrix(temp)
        x_data = temp[:, 1:10]
        y_data = temp[:, 0]

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.1).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'Year':
        temp = pd.read_table('data/Year/YearPredictionMSD.txt', header = None, sep = ',')

        temp = np.matrix(temp)
        x_data = temp[:, 1:]
        y_data = temp[:, 0]

        x_train = x_data[0:463715, :]
        y_train = y_data[0:463715]

        x_test = x_data[463715:, :]
        y_test = y_data[463715:]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)


    return x_train, y_train, x_test, y_test

