import mlrose_hiive as mlrose
import numpy as np
import cProfile
import re
import random
import helper_funcs as hf
import define_problems as dp
random.seed(123456)

# Define problem w/ 3 sizes
problem_kp_s, problem_kp_m, problem_kp_l = dp.knapsack_probs()
iteration_list = 4 * np.arange(101)

# RHC
max_attempts = np.array([10, 40, 60])
restarts = np.array([0])
run_results_RHC_s = hf.eval_5_times(mlrose.RHCRunner, 0, problem_kp_s, 'test_1', iteration_list, restarts,
                 max_attempts=max_attempts[0], generate_curves=True)
run_results_RHC_m = hf.eval_5_times(mlrose.RHCRunner, 0, problem_kp_m, 'test_1', iteration_list, restarts,
                 max_attempts=max_attempts[1], generate_curves=True)
run_results_RHC_l = hf.eval_5_times(mlrose.RHCRunner, 0, problem_kp_l, 'test_1', iteration_list, restarts,
                 max_attempts=max_attempts[2], generate_curves=True)


# Simulated Annealing
temperature_list = np.array([0.4789, 0.01, 0.8958])
run_results_SA_s = hf.eval_5_times(mlrose.SARunner, 1, problem_kp_s, 'test_1', iteration_list, temperature_list[0], decay_list=None,
                 max_attempts=50, generate_curves=True)
run_results_SA_m = hf.eval_5_times(mlrose.SARunner, 1, problem_kp_m, 'test_1', iteration_list, temperature_list[1], decay_list=None,
                 max_attempts=50, generate_curves=True)
run_results_SA_l = hf.eval_5_times(mlrose.SARunner, 1, problem_kp_l, 'test_1', iteration_list, temperature_list[2], decay_list=None,
                 max_attempts=50, generate_curves=True)


# Genetic algorithm
population_sizes = np.array([5, 80, 260])
mutation_rates = [0.0062]
run_results_GA_s = hf.eval_5_times(mlrose.GARunner, 2, problem_kp_s, 'test_1', iteration_list, population_sizes[0], mutation_rates[0],
                 max_attempts=50, generate_curves=True)
run_results_GA_m = hf.eval_5_times(mlrose.GARunner, 2, problem_kp_m, 'test_1', iteration_list, population_sizes[1], mutation_rates[0],
                 max_attempts=50, generate_curves=True)
run_results_GA_l = hf.eval_5_times(mlrose.GARunner, 2, problem_kp_l, 'test_1', iteration_list, population_sizes[2], mutation_rates[0],
                 max_attempts=50, generate_curves=True)


# MIMIC
population_sizes = np.array([20, 180, 440])
keep_percentages = np.array([0.01, 0.2705, 0.218])
run_results_MIMIC_s = hf.eval_5_times(mlrose.MIMICRunner, 2, problem_kp_s, 'test_1', iteration_list, population_sizes[0], keep_percentages[0],
                 max_attempts=50, generate_curves=True, use_fast_mimic=True)
run_results_MIMIC_m = hf.eval_5_times(mlrose.MIMICRunner, 2, problem_kp_m, 'test_1', iteration_list, population_sizes[1], keep_percentages[1],
                 max_attempts=50, generate_curves=True, use_fast_mimic=True)
run_results_MIMIC_l = hf.eval_5_times(mlrose.MIMICRunner, 2, problem_kp_l, 'test_1', iteration_list, population_sizes[2], keep_percentages[2],
                 max_attempts=50, generate_curves=True, use_fast_mimic=True)


# Plot comparison
hf.compare_algo_plots_and_size(run_results_RHC_s, run_results_RHC_m, run_results_RHC_l, run_results_SA_s, run_results_SA_m,
                                run_results_SA_l, run_results_GA_s, run_results_GA_m, run_results_GA_l, run_results_MIMIC_s,
                                run_results_MIMIC_m, run_results_MIMIC_l, 'Knapsack Size Fitness Comparison')
hf.compare_algo_plots(run_results_RHC_m, run_results_SA_m, run_results_GA_m, run_results_MIMIC_m, 'Knapsack Algo Comparison')