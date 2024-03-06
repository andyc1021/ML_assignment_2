import mlrose_hiive as mlrose
import numpy as np
import cProfile
import re
import random
import matplotlib.pyplot as plt
import math

random.seed(123456)


# Function to evaluate 5 times to account for randomness

def eval_5_times(runner_func, is_diff, *args, **kwargs):
    results = dict()
    feval_ = np.empty((5,args[3].size * (args[2].size)))
    fitness_ = np.empty((5,args[3].size * (args[2].size)))
    time_ = np.empty((5,args[3].size * (args[2].size)))
    iterations = np.empty((5,args[3].size * (args[2].size)))
    for i in range(1, 6): # seeds 1-5
        if is_diff == 1:
            run_setup = runner_func(args[0], args[1], i, args[2], [args[3]], **kwargs)
        elif is_diff == 2:
            run_setup = runner_func(args[0], args[1], i, args[2], [args[3]], [args[4]], **kwargs)
        else:
            run_setup = runner_func(args[0], args[1], i, *args[2:], **kwargs)
        df_run_stats, df_run_curves = run_setup.run()
        feval_[i-1] = df_run_stats['FEvals'].values
        fitness_[i-1] = df_run_stats['Fitness'].values
        time_[i-1] = df_run_stats['Time'].values
        iterations[i-1] = df_run_stats['Iteration'].values

    results['avg_feval_'] = np.average(feval_, axis=0)
    results['avg_fitness_'] = np.average(fitness_, axis=0)
    results['avg_time_'] = np.average(time_, axis=0)
    results['std_feval_'] = np.std(feval_, axis=0)
    results['std_fitness_'] = np.std(fitness_, axis=0)
    results['std_time_'] = np.std(time_, axis=0)
    results['iterations'] = iterations[0]

    return results

def tune_3_size(runner_func, problem_s, problem_m, problem_l, tune_array, *args, **kwargs):
    tuned_hyp = np.empty((3,1))
    max_fitness = np.empty((3, 1))
    fitness_s = np.empty((5, (args[1].size)))
    fitness_m = np.empty((5, (args[1].size)))
    fitness_l = np.empty((5, (args[1].size)))
    max_fitness_s = 0
    max_fitness_m = 0
    max_fitness_l = 0
    for j in range(tune_array.size):
        for i in range(1, 6): # seeds 1-5
            run_setup_s = runner_func(problem_s, args[0], i, args[1], [tune_array[j]], **kwargs)
            run_setup_m = runner_func(problem_m, args[0], i, args[1], [tune_array[j]], **kwargs)
            run_setup_l = runner_func(problem_l, args[0], i, args[1], [tune_array[j]], **kwargs)
            df_run_stats_s = run_setup_s.run()[0]
            df_run_stats_m = run_setup_m.run()[0]
            df_run_stats_l = run_setup_l.run()[0]
            fitness_s[i-1] = df_run_stats_s['Fitness'].values
            fitness_m[i - 1] = df_run_stats_m['Fitness'].values
            fitness_l[i - 1] = df_run_stats_l['Fitness'].values
        avg_fitness_s = np.average(fitness_s[:,-1])
        avg_fitness_m = np.average(fitness_m[:, -1])
        avg_fitness_l = np.average(fitness_l[:, -1])
        if avg_fitness_s > max_fitness_s:
            max_fitness_s = avg_fitness_s
            tuned_hyp[0] = tune_array[j]
        if avg_fitness_m > max_fitness_m:
            max_fitness_m = avg_fitness_m
            tuned_hyp[1] = tune_array[j]
        if avg_fitness_l > max_fitness_l:
            max_fitness_l = avg_fitness_l
            tuned_hyp[2] = tune_array[j]
    max_fitness[0] = max_fitness_s
    max_fitness[1] = max_fitness_m
    max_fitness[2] = max_fitness_l

    return tuned_hyp, max_fitness

def tune_3_size_RHC(runner_func, problem_s, problem_m, problem_l, tune_array, *args, **kwargs):
    tuned_hyp = np.empty((3,1))
    max_fitness = np.empty((3, 1))
    fitness_s = np.empty((5, (args[1].size)))
    fitness_m = np.empty((5, (args[1].size)))
    fitness_l = np.empty((5, (args[1].size)))
    max_fitness_s = 0
    max_fitness_m = 0
    max_fitness_l = 0
    for j in range(tune_array.size):
        for i in range(1, 6): # seeds 1-5
            run_setup_s = runner_func(problem_s, args[0], i, args[1], [0], max_attempts=tune_array[j], **kwargs)
            run_setup_m = runner_func(problem_m, args[0], i, args[1], [0], max_attempts=tune_array[j], **kwargs)
            run_setup_l = runner_func(problem_l, args[0], i, args[1], [0], max_attempts=tune_array[j], **kwargs)
            df_run_stats_s = run_setup_s.run()[0]
            df_run_stats_m = run_setup_m.run()[0]
            df_run_stats_l = run_setup_l.run()[0]
            fitness_s[i-1] = df_run_stats_s['Fitness'].values
            fitness_m[i - 1] = df_run_stats_m['Fitness'].values
            fitness_l[i - 1] = df_run_stats_l['Fitness'].values
        avg_fitness_s = np.average(fitness_s[:,-1])
        avg_fitness_m = np.average(fitness_m[:, -1])
        avg_fitness_l = np.average(fitness_l[:, -1])
        if avg_fitness_s > max_fitness_s:
            max_fitness_s = avg_fitness_s
            tuned_hyp[0] = tune_array[j]
        if avg_fitness_m > max_fitness_m:
            max_fitness_m = avg_fitness_m
            tuned_hyp[1] = tune_array[j]
        if avg_fitness_l > max_fitness_l:
            max_fitness_l = avg_fitness_l
            tuned_hyp[2] = tune_array[j]
    max_fitness[0] = max_fitness_s
    max_fitness[1] = max_fitness_m
    max_fitness[2] = max_fitness_l


    return tuned_hyp, max_fitness

def tune_3_size_v2(runner_func, problem_s, problem_m, problem_l, hyp_switch, tune_array, *args, **kwargs):
    tuned_hyp = np.empty((3,1))
    max_fitness = np.empty((3,1))
    fitness_s = np.empty((5, (args[1].size)))
    fitness_m = np.empty((5, (args[1].size)))
    fitness_l = np.empty((5, (args[1].size)))
    max_fitness_s = 0
    max_fitness_m = 0
    max_fitness_l = 0
    for j in range(tune_array.size):
        for i in range(1, 6): # seeds 1-5
            if hyp_switch == 0:
                run_setup_s = runner_func(problem_s, args[0], i, args[1], [tune_array[j]], args[2], **kwargs)
                run_setup_m = runner_func(problem_m, args[0], i, args[1], [tune_array[j]], args[2], **kwargs)
                run_setup_l = runner_func(problem_l, args[0], i, args[1], [tune_array[j]], args[2], **kwargs)
            else:
                run_setup_s = runner_func(problem_s, args[0], i, args[1],  args[2][0], [tune_array[j]], **kwargs)
                run_setup_m = runner_func(problem_m, args[0], i, args[1],  args[2][1], [tune_array[j]] ,**kwargs)
                run_setup_l = runner_func(problem_l, args[0], i, args[1],  args[2][2], [tune_array[j]] ,**kwargs)
            df_run_stats_s = run_setup_s.run()[0]
            df_run_stats_m = run_setup_m.run()[0]
            df_run_stats_l = run_setup_l.run()[0]
            fitness_s[i-1] = df_run_stats_s['Fitness'].values
            fitness_m[i - 1] = df_run_stats_m['Fitness'].values
            fitness_l[i - 1] = df_run_stats_l['Fitness'].values
        avg_fitness_s = np.average(fitness_s[:,-1])
        avg_fitness_m = np.average(fitness_m[:, -1])
        avg_fitness_l = np.average(fitness_l[:, -1])
        if avg_fitness_s > max_fitness_s:
            max_fitness_s = avg_fitness_s
            tuned_hyp[0] = tune_array[j]
        if avg_fitness_m > max_fitness_m:
            max_fitness_m = avg_fitness_m
            tuned_hyp[1] = tune_array[j]
        if avg_fitness_l > max_fitness_l:
            max_fitness_l = avg_fitness_l
            tuned_hyp[2] = tune_array[j]
    max_fitness[0] = max_fitness_s
    max_fitness[1] = max_fitness_m
    max_fitness[2] = max_fitness_l

    return tuned_hyp, max_fitness


def plot_results(results, title_s):
    # fitness vs iter
    plt.figure()
    plt.plot(results['iterations'], results['avg_fitness_'], label='avg')
    plt.plot(results['iterations'], results['avg_fitness_'] + results['std_fitness_'], '--r', label='1-sigma')
    plt.plot(results['iterations'], results['avg_fitness_'] - results['std_fitness_'], '--r')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(title_s)
    plt.legend()

    # function eval
    plt.figure()
    plt.plot(results['iterations'], results['avg_feval_'], label='avg')
    plt.plot(results['iterations'], results['avg_feval_'] + results['std_feval_'], '--r', label='1-sigma')
    plt.plot(results['iterations'], results['avg_feval_'] - results['std_feval_'], '--r')
    plt.xlabel('Iterations')
    plt.ylabel('Function Evaluation')
    plt.title(title_s)
    plt.legend()

    # wall time
    plt.figure()
    plt.plot(results['iterations'], results['avg_time_'], label='avg')
    plt.plot(results['iterations'], results['avg_time_'] + results['std_time_'], '--r', label='1-sigma')
    plt.plot(results['iterations'], results['avg_time_'] - results['std_time_'], '--r')
    plt.xlabel('Iterations')
    plt.ylabel('Wall Time (s)')
    plt.title(title_s)
    plt.legend()
    plt.show()

def compare_algo_plots(results_rhc, results_sa, results_ga, results_mimic, title_s):
    # fitness vs iter
    plt.figure()
    plt.plot(results_rhc['iterations'], results_rhc['avg_fitness_'], '-k', label='RHC')
    plt.fill_between(results_rhc['iterations'], results_rhc['avg_fitness_'] - results_rhc['std_fitness_'], results_rhc['avg_fitness_'] + results_rhc['std_fitness_'], color='lightgray', alpha=0.3)
    plt.plot(results_sa['iterations'], results_sa['avg_fitness_'], '-r', label='SA')
    plt.fill_between(results_sa['iterations'], results_sa['avg_fitness_'] - results_sa['std_fitness_'], results_sa['avg_fitness_'] + results_sa['std_fitness_'], color='red', alpha=0.3)
    plt.plot(results_ga['iterations'], results_ga['avg_fitness_'], '-g', label='GA')
    plt.fill_between(results_ga['iterations'], results_ga['avg_fitness_'] - results_ga['std_fitness_'],
                     results_ga['avg_fitness_'] + results_ga['std_fitness_'], color='green', alpha=0.3)
    plt.plot(results_mimic['iterations'], results_mimic['avg_fitness_'], '-b', label='MIMIC')
    plt.fill_between(results_mimic['iterations'], results_mimic['avg_fitness_'] - results_mimic['std_fitness_'],
                     results_mimic['avg_fitness_'] + results_mimic['std_fitness_'], color='blue', alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(title_s)
    plt.legend()
    save_str = title_s + 'fit_iter.png'
    plt.savefig(save_str, bbox_inches='tight')

    plt.figure()
    plt.plot(results_rhc['avg_time_'], results_rhc['avg_fitness_'], '-k', label='RHC')
    plt.fill_between(results_rhc['avg_time_'], results_rhc['avg_fitness_'] - results_rhc['std_fitness_'],
                     results_rhc['avg_fitness_'] + results_rhc['std_fitness_'], color='lightgray', alpha=0.3)
    plt.plot(results_sa['avg_time_'], results_sa['avg_fitness_'], '-r', label='SA')
    plt.fill_between(results_sa['avg_time_'], results_sa['avg_fitness_'] - results_sa['std_fitness_'],
                     results_sa['avg_fitness_'] + results_sa['std_fitness_'], color='red', alpha=0.3)
    plt.plot(results_ga['avg_time_'], results_ga['avg_fitness_'], '-g', label='GA')
    plt.fill_between(results_ga['avg_time_'], results_ga['avg_fitness_'] - results_ga['std_fitness_'],
                     results_ga['avg_fitness_'] + results_ga['std_fitness_'], color='green', alpha=0.3)
    plt.plot(results_mimic['avg_time_'], results_mimic['avg_fitness_'], '-b', label='MIMIC')
    plt.fill_between(results_mimic['avg_time_'], results_mimic['avg_fitness_'] - results_mimic['std_fitness_'],
                     results_mimic['avg_fitness_'] + results_mimic['std_fitness_'], color='blue', alpha=0.3)
    plt.xlabel('Wall Time (s)')
    plt.ylabel('Fitness')
    plt.title(title_s)
    plt.legend()
    save_str = title_s + 'fit_time.png'
    plt.savefig(save_str, bbox_inches='tight')

    plt.figure()
    plt.plot(results_rhc['iterations'], results_rhc['avg_feval_'], '-k', label='RHC')
    plt.fill_between(results_rhc['iterations'], results_rhc['avg_feval_'] - results_rhc['std_feval_'],
                     results_rhc['avg_feval_'] + results_rhc['std_feval_'], color='lightgray', alpha=0.3)
    plt.plot(results_sa['iterations'], results_sa['avg_feval_'], '-r', label='SA')
    plt.fill_between(results_sa['iterations'], results_sa['avg_feval_'] - results_sa['std_feval_'],
                     results_sa['avg_feval_'] + results_sa['std_feval_'], color='red', alpha=0.3)
    plt.plot(results_ga['iterations'], results_ga['avg_feval_'], '-g', label='GA')
    plt.fill_between(results_ga['iterations'], results_ga['avg_feval_'] - results_ga['std_feval_'],
                     results_ga['avg_feval_'] + results_ga['std_feval_'], color='green', alpha=0.3)
    plt.plot(results_mimic['iterations'], results_mimic['avg_feval_'], '-b', label='MIMIC')
    plt.fill_between(results_mimic['iterations'], results_mimic['avg_feval_'] - results_mimic['std_feval_'],
                     results_mimic['avg_feval_'] + results_mimic['std_feval_'], color='blue', alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel('fevals')
    plt.title(title_s)
    plt.legend()
    save_str = title_s + 'iter_feval.png'
    plt.savefig(save_str, bbox_inches='tight')

    plt.figure()
    labels = ['RHC', 'SA', 'GA', 'MIMIC']
    feval_iter = [results_rhc['avg_feval_'][-1] / results_rhc['iterations'][-1], results_sa['avg_feval_'][-1] / results_sa['iterations'][-1],
                  results_ga['avg_feval_'][-1] / results_ga['iterations'][-1], results_mimic['avg_feval_'][-1] / results_mimic['iterations'][-1]]
    x = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots()
    bars1 = ax.bar(x, feval_iter, width, label='fevals/iter')
    ax.set_ylabel('fevals/iteration')
    ax.set_title(title_s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    save_str = title_s + 'feval_iter.png'
    plt.savefig(save_str, bbox_inches='tight')
    plt.show()

def compare_algo_plots_and_size(rhc_s, rhc_m, rhc_l, sa_s, sa_m, sa_l, ga_s, ga_m, ga_l, mi_s, mi_m, mi_l, title_s):
    labels = ['Small', 'Medium', 'Large']
    rhc_vals = [rhc_s['avg_fitness_'][-1], rhc_m['avg_fitness_'][-1], rhc_l['avg_fitness_'][-1]]
    sa_vals = [sa_s['avg_fitness_'][-1], sa_m['avg_fitness_'][-1], sa_l['avg_fitness_'][-1]]
    ga_vals = [ga_s['avg_fitness_'][-1], ga_m['avg_fitness_'][-1], ga_l['avg_fitness_'][-1]]
    mim_vals = [mi_s['avg_fitness_'][-1], mi_m['avg_fitness_'][-1], mi_l['avg_fitness_'][-1]]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - 1.5 * width, rhc_vals, width, label='RHC')
    bars2 = ax.bar(x - 0.5 * width, sa_vals, width, label='SA')
    bars3 = ax.bar(x + 0.5 * width, ga_vals, width, label='GA')
    bars4 = ax.bar(x + 1.5 * width, mim_vals, width, label='MIMIC')

    ax.set_ylabel('Fitness')
    ax.set_title(title_s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    save_str = title_s + 'size_comp.png'
    plt.savefig(save_str, bbox_inches='tight')

    plt.show()

def nnet_plots(results_rhc, results_sa, results_ga, results_control, title_s):
    # fitness vs iter
    plt.figure()
    plt.plot(results_rhc[1]['Iteration'], results_rhc[1]['Fitness'], '-k', label='RHC')
    plt.plot(results_sa[1]['Iteration'], results_sa[1]['Fitness'], '-r', label='SA')
    plt.plot(results_ga[1]['Iteration'], results_ga[1]['Fitness'], '-g', label='GA')
    plt.plot(results_control[1]['Iteration'], results_control[1]['Fitness'], '-b', label='Control')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(title_s)
    plt.legend()

    plt.figure()
    plt.plot(results_rhc[1]['Time'], results_rhc[1]['Fitness'], '-k', label='RHC')
    plt.plot(results_sa[1]['Time'], results_sa[1]['Fitness'], '-r', label='SA')
    plt.plot(results_ga[1]['Time'], results_ga[1]['Fitness'], '-g', label='GA')
    plt.plot(results_control[1]['Time'], results_control[1]['Fitness'], '-b', label='Control')
    plt.xlabel('Wall Time (s)')
    plt.ylabel('Fitness')
    plt.title(title_s)
    plt.legend()
    plt.show()


def generate_edges(num_edges):
    edges = []
    i = 0
    while i < num_edges:
        test_point = (random.randint(0, math.ceil(num_edges / 2)), random.randint(0, math.ceil(num_edges / 2)))
        if test_point not in edges:
            edges.append(test_point)
            i += 1
    return edges


def generate_objs(num_objs):
    weights = []
    values = []
    for i in range(num_objs):
        weights.append(random.randint(1, 20))
        values.append(random.randint(1, 20))
    return weights, values