"""
Generates configuration files for  genetic algorithm implementation,
using configparser library. Creates directory, saves script
and runs it. Takes number of couplings (even numbers) and
directory name as argument.
"""

import configparser
import os
import sys

# create directory to save configuration, scripts and results
directory = sys.argv[2]
script = "average.py"
module = "gmod.py"

number_of_samples = 10  # number of times the experiment is repeated

# creates instance of ConfigParser
config = configparser.ConfigParser()

# system parameters
n = int(sys.argv[1])
number_of_couplings = n - 1
delta = 1.0
transmission_time = n



# genetic algorithm parameters (used by PyGAD library, see Documentation)
num_generations = 2000
sol_per_pop = 1000
maxj = n
init_range_low = 1.0
init_range_high = maxj
fidelity_tolerance = 0.99
saturation = 20
smooth_solution = True # mark as true to add smoothing factor to fitness

# if using smooth_solution, configure smoothing weight (if not these parameters are ignored)
beta_is_gene = False  # if True, take weight factor as gene
beta = 0.9  # else, fix weight factor
num_genes = number_of_couplings // 2 + (1 - n % 2) + 1 * beta_is_gene

# crossover and parent selection (used by PyGAD library, see Documentation)
num_parents_mating = sol_per_pop // 5
parent_selection_type = "sss"
keep_elitism = sol_per_pop // 10
crossover_type = "uniform"
crossover_probability = 0.6

# general mutation parameters
mutation_probability = 0.99
mutation_num_genes = 1

# adaptive mutation parameters
weak_change = 0.03  # 3% mutation after solution converges
strong_change_first = 0.1  # 10% percent mutation for first chain element
strong_change_rest = 0.05  # 5% mutation for the rest of the chain

# on generation parameters
og_print = True  # Print best solution, fidelity and fitness every generation
check_tol = True  # Check if fidelity reaches certain value every generation
histogram = False  # Exports a histogram with the population's fitness

# generation of config file
config["saving_and_running"] = {
    "directory": directory,
    "script": script,
    "module": module,
    "number_of_samples": str(number_of_samples),
}

config["system_parameters"] = {
    "number_of_couplings": str(number_of_couplings),
    "n": str(n),
    "delta": str(delta),
    "transmission_time": str(transmission_time),
}

config["ga_initialization"] = {
    "num_generations": str(num_generations),
    "num_genes": str(num_genes),
    "sol_per_pop": str(sol_per_pop),
    "maxj": str(maxj),
    "init_range_low": str(init_range_low),
    "init_range_high": str(init_range_high),
    "fidelity_tolerance": str(fidelity_tolerance),
    "saturation": str(saturation),
    "smooth_solution": str(smooth_solution),
    "beta_is_gene": str(beta_is_gene),
    "beta": str(beta)
}

config["parent_selection"] = {
    "num_parents_mating": str(num_parents_mating),
    "parent_selection_type": parent_selection_type,
    "keep_elitism": str(keep_elitism),
}

config["crossover"] = {
    "crossover_type": crossover_type,
    "crossover_probability": str(crossover_probability),
}

config["mutation"] = {
    "mutation_probability": str(mutation_probability),
    "mutation_num_genes": str(mutation_num_genes),
}

config["adaptivemutation"] = {
    "weak_change": str(weak_change),
    "strong_change_first": str(strong_change_first),
    "strong_change_rest": str(strong_change_rest),
}

config["on_generation"] = {
    "og_print": str(og_print),
    "check_tol": str(check_tol),
    "histogram": str(histogram),
}

# create directory, generate and save config file to directory
# (different file for each number of couplings)

isExist = os.path.exists(directory)

if not isExist:
    os.mkdir(directory)
else:
    print("Warning: Directory already exists")

config_name = directory + "/ga" + str(n) + ".ini"
with open(config_name, "w") as configfile:
    config.write(configfile)

# copy script and module to directory
src = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.join(src, script)
mod_name = os.path.join(src, module)
cmd = f'cp "{script_name}" "{directory}"'
os.system(cmd)
cmd = f'cp "{mod_name}" "{directory}"'
os.system(cmd)

# run script from inside directory
script_name = directory + "/" + script
cmd = f'python3 "{script_name}" "{config_name}"'
os.system(cmd)
