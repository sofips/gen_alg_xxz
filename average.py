"""
Script used to run genetic algorithm experiments for the XXZ
Hamiltonian. Takes config file as argument and reads parameters
from it. Executes n_samples runs of genetic algorithm using
Pygad library.

Saves (in the specified directory):
    - fitness history for every sample
    - solution (couplings) obtained for every sample
    - summary file with rows: dimension, delta, final fidelity, CPU time
    and generations (for every sample)


"""

from gmod import *
import csv
import pygad
import sys
import time
import os
import configparser


# get parameters from config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = str(sys.argv[1])
print(initfile)
config = configparser.ConfigParser()
config.read(initfile)

# saving and running
dirname = config.get("saving_and_running", "directory")
n_samples = config.getint("saving_and_running", "number_of_samples")

# system parameters
nj = config.getint("system_parameters", "number_of_couplings")
n = config.getint("system_parameters", "n")
delta = config.getfloat("system_parameters", "delta")
transmission_time = config.getfloat("system_parameters", "transmission_time")
beta_is_gene = config.getboolean("system_parameters", "beta_is_gene")
beta = config.getfloat("system_parameters", "beta")

fidelity_args = [n, delta, transmission_time, beta_is_gene, beta]


# genetic algorithm parameters
num_generations = config.getint("ga_initialization", "num_generations")
num_genes = config.getint("ga_initialization", "num_genes")
sol_per_pop = config.getint("ga_initialization", "sol_per_pop")
maxj = config.getfloat("ga_initialization", "maxj")
init_range_low = config.getfloat("ga_initialization", "init_range_low")
init_range_high = config.getfloat("ga_initialization", "init_range_high")
fidelity_tolerance = config.getfloat("ga_initialization", "fidelity_tolerance")
saturation = config.getint("ga_initialization", "saturation")

# crossover and parent selection

num_parents_mating = config.getint("parent_selection", "num_parents_mating")
parent_selection_type = config.get("parent_selection", "parent_selection_type")
keep_elitism = config.getint("parent_selection", "keep_elitism")
crossover_type = config.get("crossover", "crossover_type")
crossover_probability = config.getfloat("crossover", "crossover_probability")

# mutation
mutation_probability = config.getfloat("mutation", "mutation_probability")
mutation_num_genes = config.getint("mutation", "mutation_num_genes")

# specific parameters for selected mutation

mutation_func = adaptivemut

weak_change = config.getfloat("adaptivemutation", "weak_change")
strong_change_first = config.getfloat("adaptivemutation", "strong_change_first")
strong_change_rest = config.getfloat("adaptivemutation", "strong_change_rest")

mutation_args = [weak_change, strong_change_first, strong_change_rest]

# on generation parameters
og_print = config.getboolean("on_generation", "og_print")
check_tol = config.getboolean("on_generation", "check_tol")
histogram = config.getboolean("on_generation", "histogram")

on_generation_parameters = [
    n,
    maxj,
    transmission_time,
    delta,
    fidelity_tolerance,
    check_tol,
    og_print,
    dirname,
    beta_is_gene,
    histogram,
]

# call construction functions
on_generation = generation_func_constructor(generation_func, on_generation_parameters)
fitness_func = fitness_func_constructor(j_fidelity, fidelity_args)
mutation_type = mutation_func_constructor(adaptivemut, mutation_args)

genespace = generate_gsp1(n, maxj, beta_is_gene)
stop_criteria = ["saturate_" + str(saturation), "reach_" + str(fidelity_tolerance)]


filename = dirname + "/nvsmaxfid.dat"

with open(filename, "a") as f:
    for i in range(n_samples):
        writer = csv.writer(f, delimiter=" ")

        solutions_fname = dirname + "/jn" + str(n) + "sample" + str(i) + ".dat"
        fitness_history_fname = (
            dirname + "/fitness_history_n" + str(n) + "_sample" + str(i) + ".dat"
        )

        t1 = time.time()

        initial_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            parent_selection_type=parent_selection_type,
            keep_elitism=keep_elitism,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            gene_space=genespace,
            mutation_type=mutation_type,
            on_generation=on_generation,
            mutation_num_genes=mutation_num_genes,
            mutation_probability=mutation_probability,
            stop_criteria=stop_criteria,
        )

        initial_instance.run()

        solution, solution_fitness, solution_idx = initial_instance.best_solution()

        t2 = time.time()
        trun = t2 - t1
        maxg = initial_instance.generations_completed

        final_fidelity = fidelity(solution, n, delta, transmission_time, beta_is_gene)

        row = [
            nj,
            "{:.8f}".format(delta),
            "{:.8f}".format(final_fidelity),
            "{:.8f}".format(trun),
            maxg,
        ]
        writer.writerow(row)

        couplings_to_file(solution, solutions_fname, "w")
        fitness_history_to_file(initial_instance, fitness_history_fname)
