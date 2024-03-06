import mlrose_hiive as mlrose
import numpy as np
import cProfile
import re
import random
import helper_funcs as hf
random.seed(123456)

# Six Peaks
def six_P_probs():
    num_states_s = 5
    fitness_6P = mlrose.SixPeaks(t_pct=0.15)
    problem_6P_s = mlrose.DiscreteOpt(length = num_states_s, fitness_fn = fitness_6P, maximize = True)

    num_states_m = 20
    problem_6P_m = mlrose.DiscreteOpt(length = num_states_m, fitness_fn = fitness_6P, maximize = True)

    num_states_l = 50
    problem_6P_l = mlrose.DiscreteOpt(length = num_states_l, fitness_fn = fitness_6P, maximize = True)

    return problem_6P_s, problem_6P_m, problem_6P_l

def k_color_probs():
    # K Color
    num_edges_s = 5
    edges_s = hf.generate_edges(num_edges_s)
    fitness_kc_s = mlrose.MaxKColor(edges_s, maximize = True)
    problem_kc_s = mlrose.DiscreteOpt(length = num_edges_s+1, fitness_fn = fitness_kc_s, maximize = True)

    num_edges_m = 20
    edges_m = hf.generate_edges(num_edges_m)
    fitness_kc_m = mlrose.MaxKColor(edges_m, maximize = True)
    problem_kc_m = mlrose.DiscreteOpt(length = num_edges_m+1, fitness_fn = fitness_kc_m, maximize = True)

    num_edges_l = 50
    edges_l = hf.generate_edges(num_edges_l)
    fitness_kc_l = mlrose.MaxKColor(edges_l, maximize = True)
    problem_kc_l = mlrose.DiscreteOpt(length = num_edges_l+1, fitness_fn = fitness_kc_l, maximize = True)

    return problem_kc_s, problem_kc_m, problem_kc_l

def knapsack_probs():
    # Knapsack
    max_weight_pct = 0.6
    num_obj_s = 5
    weights_s, values_s = hf.generate_objs(num_obj_s)
    fitness_kn_s = mlrose.Knapsack(weights_s, values_s, max_weight_pct)
    problem_kn_s = mlrose.DiscreteOpt(length = num_obj_s, fitness_fn = fitness_kn_s, maximize = True)

    num_obj_m = 20
    weights_m, values_m = hf.generate_objs(num_obj_m)
    fitness_kn_m = mlrose.Knapsack(weights_m, values_m, max_weight_pct)
    problem_kn_m = mlrose.DiscreteOpt(length = num_obj_m, fitness_fn = fitness_kn_m, maximize = True)

    num_obj_l = 50
    weights_l, values_l = hf.generate_objs(num_obj_l)
    fitness_kn_l = mlrose.Knapsack(weights_l, values_l, max_weight_pct)
    problem_kn_l = mlrose.DiscreteOpt(length = num_obj_l, fitness_fn = fitness_kn_l, maximize = True)

    return problem_kn_s, problem_kn_m, problem_kn_l