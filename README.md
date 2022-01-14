A Kernel-Expanded Stochastic Neural Network
===============================================================

### Package Requirements
Requirements for Pytorch see [Pytorch](http://pytorch.org/) for installation instructions

Requirements for thundersvm see https://github.com/Xtra-Computing/thundersvm for installation instructions


### Data Preparation

For real data example, data needs to be downloaded and stored in the "./data/data_name/" folder(replace data_name by the name of the data set). Please refer to process_data.py for more details.

### How to run
kstonet.py uses thundersvm package to implement SVR, which uses GPU to accelerate computaion. It is suitable to be used for relatively large data set when there are GPUs available.

kstonet_parallel.py uses sklearn package to implement SVR, SVRs are solved in parallel. It is suitable to be used on a machine with multiple CPUs but no GPUs.

Experiments on different data set with different parameters can be done commands like: 
```{python}
python kstonet.py --data_name "data_set_name" --parameter parameter_value
```
Some commands to run experiments in the paper are given below. For completeness, we also include the command to run experiments for KNN model.
#### A full row rank example:
```{python}
python kstonet.py --data_name 'full_row_rank' --layer 1 --unit 5 --sigma 0.001 --C 1 --epsilon 0.1 --lr 0.0000005 --nepoch 40 --seed 0
python kstonet.py --data_name 'global_optimal' --layer 1 --unit 5 --sigma 0.01 --C 5 --epsilon 0.01 --lr 0.0005 --nepoch 40 --seed 0
python knn.py --data_name 'full_row_rank' --layer 1 --unit 5 --lr 0.005 --nepoch 2000 --batch_train 100 --lasso 0 --seed 0
python knn.py --data_name 'global_optimal' --layer 1 --unit 5 --lr 0.01 --nepoch 1000 --batch_train 100 --lasso 0 --seed 0
```
#### A measurement error example:
```{python}
python kstonet.py --data_name 'measurement_error' --layer 1 --unit 5 --sigma 0.01 --C 1 --epsilon 0.05 --lr 0.00005 --alpha 0.1 --nepoch 1000 --model_path 'test_run_one_layer/' --seed 0
python kstonet.py --data_name 'measurement_error' --layer 3 --unit 20 20 20 --sigma 0.001 0.001 0.01 --C 1 --epsilon 0.05 --lr 0.00005 --alpha 0.1 --nepoch 1000 --model_path 'test_run_three_layer/' --seed 0
python knn.py --data_name 'measurement_error' --layer 1 --unit 5  --lr 0.005 --momentum 0.9 --nepoch 1000 --batch_train 100 --lasso 0 --model_path 'knn_test_run_one_layer/' --seed 0
python knn.py --data_name 'measurement_error' --layer 3 --unit 20 20 20 --lr 0.005 --momentum 0.9 --nepoch 1000 --batch_train 100 --lasso 0 --model_path 'knn_test_run_three_layer/' --seed 0
```

#### QSAR Androgen Receptor
```{python}
python kstonet.py --data_name 'qsar' --layer 1 --unit 5 --sigma 0.001 --C 1 --epsilon 0.1 --lr 0.00005 --alpha 0.1 --nepoch 40 --regression_flag 0 --cross_validate 0
python knn.py --data_name 'qsar' --layer 1 --unit 5 --lr 0.001 --nepoch 1000 --regression_flag 0 --cross_validate 0 --lasso 0.0001
```


#### CoverType
For CoverType experiments, we can use kstonet.py to run the experiment similar to the examples above. 

But for this data set, the number of training samples are very large, it is recommended to solve SVRs separately. The following command solves the SVR corresponds to the first hidden unit in the first layer. The solve_index specifies which SVR to solve.
```{python}
python kstonet_covertype_split.py --layer 1 --unit 50 --load_epoch -1 --solve_index 0
```
After running the above command for solve_index = 0, 1, ... , 49. We can run the next epoch by
```{python}
python kstonet_covertype_split.py --layer 1 --unit 50 --load_epoch 0 --solve_index 0
```

#### MNIST
```{python}
python kstonet.py --data_name 'MNIST' --layer 1 --unit 20 --sigma 0.00001 --C 10 --epsilon 0.001 --lr 0.00000005 --alpha 0.1 --nepoch 10 --regression_flag 0
```
#### More UCI Datasets
For the 10 UCI datasets experiments in section 4.4. The kstonet_cross_validate.py and knn_cross_validate.py codes randomly shuffle the data for 20 times and run 10 fold cross validation. The following command gives an example of running experiments on Boston Housing data set.
```{python}
python kstonet_cross_validate.py --data_name 'Boston' --layer 1 --unit 5 --sigma 0.01 --C_list 1 2 5 10 20 --epsilon 0.01 --lr 0.0005 --alpha 0.1 --nepoch 50 --normalize_y_flag 1
python knn_cross_validate.py --data_name "Boston" --unit 50
```

#### Confidence Interval
Run 100 expriments with the following command for the simulation data(modify --seed and --model_path to get result with different seed)
```{python}
python kstonet.py --data_name 'measurement_error' --layer 1 --unit 5 --sigma 0.001 --C 1 --epsilon 0.05 --lr 0.00005 --alpha 1 --nepoch 50 --confidence_interval_flag 1 --seed 1 --model_path 'seed1/'
```
confidence_interval.py can be used to calculate the confidence interval

