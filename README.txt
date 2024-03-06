Andy Cheng
acheng99
CS 7641 A2

How to run code and model:

Requirements:
Python 3.10+
Libraries:
Anaconda & Pandas
mlrose_hiive
numpy
random
sklearn
matplotlib
math
cprofile
re


Notes:
Tuning is done in tune_algos and hard coded in after finding right hyperparameters. Can uncomment tune_algo code and run for tuning. Neural network tuning was done in apple_learning_curves
script, but process of finding values not saved as tuning done by hand by trying various values.

Steps:
1. Download code and datasets from repo: https://github.com/andyc1021/ML_assignment_2
2. Launch Python environment with required libraries.
3. Run k_color, knapsack, and six_peaks scripts individually to produce and save off figures for each random optimization problem.
4. Run apple_learning_curves to generate the final learning curves for the neural network with the different algorithms.
5. Run apple_NN to train and test final NN models using each of the 4 algorithms. Generates a Excel file with run results.