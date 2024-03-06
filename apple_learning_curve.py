import mlrose_hiive
import numpy as np
import random
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
import nnet_fixed

random.seed(123456)
np.random.seed(123456)

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

# Set iters to test and get data at
iters = np.round(np.linspace(1, 1000, 50))
#iters = np.power(2, np.arange(8))
#iters = [1, 2, 4, 8, 16, 24, 32, 40, 50, 60, 70, 80, 90]
learning_rates = np.linspace(0.0001, 0.4, 40)


# Control Case: gradient descent
nn_gd = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='gradient_descent',
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.0001, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)

train_scores_gd, valid_scores_gd = validation_curve(
    nn_gd, train_X.values, train_Y, param_name="max_iters", param_range=iters,
    cv=5, scoring="accuracy", n_jobs=-1
)
gd_iter_train_curves = np.average(train_scores_gd, axis=1)
gd_iter_val_curves = np.average(valid_scores_gd, axis=1)

# # RHC
nn_rhc = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='random_hill_climb',
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.359, early_stopping=False,
                                       clip_max=1e+10, max_attempts=50, random_state=123456, curve=True)
train_scores_rhc, valid_scores_rhc = validation_curve(
    nn_rhc, train_X.values, train_Y, param_name="max_iters", param_range=iters,
    cv=5, scoring="accuracy", n_jobs=-1)
rhc_iter_train_curves = np.average(train_scores_rhc, axis=1)
rhc_iter_val_curves = np.average(valid_scores_rhc, axis=1)

# SA
nn_sa = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='simulated_annealing',schedule=mlrose_hiive.GeomDecay(0.001),
                                       max_iters=1000, bias=False, is_classifier=True,
                                       learning_rate=0.3077, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)

train_scores_sa, valid_scores_sa = validation_curve(
    nn_sa, train_X.values, train_Y, param_name="max_iters", param_range=iters,
    cv=5, scoring="accuracy", n_jobs=-1)
sa_iter_train_curves = np.average(train_scores_sa, axis=1)
sa_iter_val_curves = np.average(valid_scores_sa, axis=1)

# GA
nn_ga = nnet_fixed.CustomNeuralNetwork(hidden_nodes=[20], activation='relu',
                                       algorithm='genetic_alg', pop_size=60,
                                       max_iters=100, bias=False, is_classifier=True,
                                       learning_rate=0.01, early_stopping=False,
                                       clip_max=1e+10, max_attempts=500, random_state=123456, curve=True)

train_scores_ga, valid_scores_ga = validation_curve(
    nn_ga, train_X.values, train_Y, param_name="max_iters", param_range=iters,
    cv=5, scoring="accuracy", n_jobs=-1)
ga_iter_train_curves = np.average(train_scores_ga, axis=1)
ga_iter_val_curves = np.average(valid_scores_ga, axis=1)



# Plot the average fitness curves
plt.figure(figsize=(10, 6))
plt.plot(iters, gd_iter_train_curves, label='GD Train', linestyle='--', color='blue')
plt.plot(iters, gd_iter_val_curves, label='GD Validation', linestyle='--', color='orange')
plt.plot(iters, rhc_iter_train_curves, label='RHC Train', linestyle='-', color='green')
plt.plot(iters, rhc_iter_val_curves, label='RHC Validation', linestyle='-', color='red')
plt.plot(iters, sa_iter_train_curves, label='SA Train', linestyle='-.', color='purple')
plt.plot(iters, sa_iter_val_curves, label='SA Validation', linestyle='-.', color='brown')
plt.plot(iters, ga_iter_train_curves, label='GA Train', linestyle=':', color='pink')
plt.plot(iters, ga_iter_val_curves, label='GA Validation', linestyle=':', color='gray')

plt.title('Average Accuracy Curves for Different Algorithms')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('NeuralNetLearningCurve_v2.png', bbox_inches='tight')
