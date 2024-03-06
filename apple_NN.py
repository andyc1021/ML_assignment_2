import mlrose_hiive
import mlrose_hiive as mlrose
import numpy as np
import cProfile
import re
import random
import helper_funcs as hf
import define_problems as dp
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import nnet_fixed
import time
random.seed(123456)
iteration_list = 4 * np.arange(101)

def get_accuracy(Y_predict, Y_truth):
    diff = Y_predict[:,0] == Y_truth
    sum = np.sum(diff)
    accuracy = sum / len(Y_truth)
    return (accuracy)

# Load apple data set
apple_data_file = 'apple_quality.csv'
apple_data = pd.read_csv(apple_data_file)
train_set, test_set = tts(apple_data.iloc[0:4000], test_size=0.15, random_state=123456)
mapping = {'bad': 1, 'good': 0}

train_Y = train_set['Quality']
train_Y = np.array([mapping[classification] for classification in train_Y.values])
train_X = train_set[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]
norm_train_x = StandardScaler().fit_transform(train_X)
train_X = pd.DataFrame(norm_train_x, columns=train_X.columns)
test_Y = test_set['Quality']
test_Y = np.array([mapping[classification] for classification in test_Y.values])
test_X = test_set[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]
norm_train_x = StandardScaler().fit_transform(test_X)
test_X = pd.DataFrame(norm_train_x, columns=test_X.columns)

# Time code for train time and evaluation time

# Control Case: gradient descent
nn_gd = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='gradient_descent',
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.0001, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)
start_time = time.time()
nn_gd_mod = nn_gd.fit(train_X.values, train_Y)
nn_gd_pred = nn_gd_mod.predict(test_X.values)
nn_gd_acc = get_accuracy(nn_gd_pred, test_Y)
end_time = time.time()
gd_time = end_time - start_time

# RHC
nn_rhc = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='random_hill_climb',
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.359, early_stopping=False,
                                       clip_max=1e+10, max_attempts=50, random_state=123456, curve=True)
start_time = time.time()
nn_rhc_mod = nn_rhc.fit(train_X.values, train_Y)
nn_rhc_pred = nn_rhc_mod.predict(test_X.values)
nn_rhc_acc = get_accuracy(nn_rhc_pred, test_Y)
end_time = time.time()
rhc_time = end_time - start_time

# SA
nn_sa = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='simulated_annealing',schedule=mlrose_hiive.GeomDecay(0.001),
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.3077, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)
start_time = time.time()
nn_sa_mod = nn_sa.fit(train_X.values, train_Y)
nn_sa_pred = nn_sa_mod.predict(test_X.values)
nn_sa_acc = get_accuracy(nn_sa_pred, test_Y)
end_time = time.time()
sa_time = end_time - start_time

# GA
nn_ga = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='genetic_alg', pop_size=60,
                                       max_iters=100, bias=False, is_classifier=True,
                                       learning_rate=0.01, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)
start_time = time.time()
nn_ga_mod = nn_ga.fit(train_X.values, train_Y)
nn_ga_pred = nn_ga_mod.predict(test_X.values)
nn_ga_acc = get_accuracy(nn_ga_pred, test_Y)
end_time = time.time()
ga_time = end_time - start_time

# Save to table
data = {
    'Algorithm': ['GD', 'RHC', 'SA', 'GA'],
    'Accuracy': [nn_gd_acc, nn_rhc_acc, nn_sa_acc, nn_ga_acc],
    'Wall Time': [gd_time, rhc_time, sa_time, ga_time]
}
df = pd.DataFrame(data)
df.to_excel('Apple_NN_data.xlsx', index=False)