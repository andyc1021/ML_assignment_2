import mlrose_hiive as mlrose
import numpy as np
import cProfile
import re
import random
import helper_funcs as hf
import define_problems as dp
random.seed(123456)

# Tuning hyperparameters
# RHC - Restarts?
# SA - Temp
# GA - Pop size, mutation rate
# MIMIC - Pop size, keep percent

iteration_list = 4 * np.arange(26)
# Tune for six peaks s,m,l
problem_6P_s, problem_6P_m, problem_6P_l = dp.six_P_probs()
# RHC
# max_attempts = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 50)))
# RHC_hyp_6P, RHC_mf_6P = hf.tune_3_size_RHC(mlrose.RHCRunner, problem_6P_s, problem_6P_m, problem_6P_l, max_attempts, 'test_1', iteration_list, decay_list=None,
#                 generate_curves=True)
# # # SA
# temps = np.linspace(0.01, 1, 20)
# SA_hyp_6P, SA_mf_6P = hf.tune_3_size(mlrose.SARunner, problem_6P_s, problem_6P_m, problem_6P_l, temps, 'test_1', iteration_list, decay_list=None,
#                   max_attempts=50, generate_curves=True)
#
population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
mutation_rates_init = [0.01]
mutation_rates = np.linspace(0.01, 1, 20)
GA_hyp_6P, GA_mf_6P = hf.tune_3_size_v2(mlrose.GARunner, problem_6P_s, problem_6P_m, problem_6P_l,0,  population_sizes, 'test_1', iteration_list, mutation_rates_init,
                                        decay_list=None, max_attempts=50, generate_curves=True)
GA_hyp_6P2, GA_mf_6P2 = hf.tune_3_size_v2(mlrose.GARunner, problem_6P_s, problem_6P_m, problem_6P_l,1,  mutation_rates, 'test_1', iteration_list, GA_hyp_6P,
                                        decay_list=None, max_attempts=50, generate_curves=True)
#
# population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
# keep_percentages_init = [0.2]
# keep_percentages = np.linspace(0.01, 1, 20)
# MIMIC_hyp_6P, MIMIC_mf_6P = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_6P_s, problem_6P_m, problem_6P_l,0,  population_sizes, 'test_1', iteration_list, keep_percentages_init,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)
# MIMIC_hyp_6P2, MIMIC_mf_6P2 = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_6P_s, problem_6P_m, problem_6P_l,1,  keep_percentages, 'test_1', iteration_list, MIMIC_hyp_6P,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)


# # Tune for k color
problem_kc_s, problem_kc_m, problem_kc_l = dp.k_color_probs()
# RHC
max_attempts = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 50)))
RHC_hyp_kc, RHC_mf_kc = hf.tune_3_size_RHC(mlrose.RHCRunner, problem_kc_s, problem_kc_m, problem_kc_l, max_attempts, 'test_1', iteration_list, decay_list=None,
                generate_curves=True)
# # SA
# temps = np.linspace(0.01, 1, 20)
# SA_hyp_kc, SA_mf_kc = hf.tune_3_size(mlrose.SARunner, problem_kc_s, problem_kc_m, problem_kc_l, temps, 'test_1', iteration_list, decay_list=None,
#                  max_attempts=50, generate_curves=True)
#
# population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
# mutation_rates_init = [0.01]
# mutation_rates = np.linspace(0.01, 1, 20)
# GA_hyp_kc, GA_mf_kc = hf.tune_3_size_v2(mlrose.GARunner, problem_kc_s, problem_kc_m, problem_kc_l,0,  population_sizes, 'test_1', iteration_list, mutation_rates_init,
#                                         decay_list=None, max_attempts=50, generate_curves=True)
# GA_hyp_kc2, GA_mf_kc2 = hf.tune_3_size_v2(mlrose.GARunner, problem_kc_s, problem_kc_m, problem_kc_l,1,  mutation_rates, 'test_1', iteration_list, GA_hyp_kc,
#                                         decay_list=None, max_attempts=50, generate_curves=True)
#
# population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
# keep_percentages_init = [0.2]
# keep_percentages = np.linspace(0.01, 1, 20)
# MIMIC_hyp_kc, MIMIC_mf_kc = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_kc_s, problem_kc_m, problem_kc_l,0,  population_sizes, 'test_1', iteration_list, keep_percentages_init,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)
# MIMIC_hyp_kc2, MIMIC_mf_kc2 = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_kc_s, problem_kc_m, problem_kc_l,1,  keep_percentages, 'test_1', iteration_list, MIMIC_hyp_kc,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)

# Tune for knapsack
problem_kn_s, problem_kn_m, problem_kn_l = dp.knapsack_probs()
# RHC
max_attempts = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 50)))
RHC_hyp_kn, RHC_mf_kn = hf.tune_3_size_RHC(mlrose.RHCRunner, problem_kn_s, problem_kn_m, problem_kn_l, max_attempts, 'test_1', iteration_list, decay_list=None,
                generate_curves=True)
# # SA
# temps = np.linspace(0.01, 1, 20)
# SA_hyp_kn, SA_mf_kn = hf.tune_3_size(mlrose.SARunner, problem_kn_s, problem_kn_m, problem_kn_l, temps, 'test_1', iteration_list, decay_list=None,
#                  max_attempts=50, generate_curves=True)
#
# population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
# mutation_rates_init = [0.1]
# mutation_rates = np.linspace(0.01, 1, 20)
# GA_hyp_kn, GA_mf_kn = hf.tune_3_size_v2(mlrose.GARunner, problem_kn_s, problem_kn_m, problem_kn_l,0,  population_sizes, 'test_1', iteration_list, mutation_rates_init,
#                                         decay_list=None, max_attempts=50, generate_curves=True)
# GA_hyp_kn2, GA_mf_kn2 = hf.tune_3_size_v2(mlrose.GARunner, problem_kn_s, problem_kn_m, problem_kn_l,1,  mutation_rates, 'test_1', iteration_list, GA_hyp_kn,
#                                         decay_list=None, max_attempts=50, generate_curves=True)
#
# population_sizes = np.concatenate((np.array([2, 5, 10]), 20 * np.arange(1, 25)))
# keep_percentages_init = [0.2]
# keep_percentages = np.linspace(0.01, 1, 20)
# MIMIC_hyp_kn, MIMIC_mf_kn = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_kn_s, problem_kn_m, problem_kn_l,0,  population_sizes, 'test_1', iteration_list, keep_percentages_init,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)
# MIMIC_hyp_kn2, MIMIC_mf_kn2 = hf.tune_3_size_v2(mlrose.MIMICRunner, problem_kn_s, problem_kn_m, problem_kn_l,1,  keep_percentages, 'test_1', iteration_list, MIMIC_hyp_kn,
#                                         decay_list=None, max_attempts=50, generate_curves=True, use_fast_mimic=True)

a = 1