"""
Genetic algorithm related functions. Fidelity for Heisenberg Hamiltonians.
"""
import math
import random
import datetime
import numpy as np
import scipy.linalg as la
import csv
import os
import matplotlib.pyplot as plt

# Mutation functions


def mutation2(offspring, change_first, change_rest, beta_is_gene=False):
    """
    Parameters:
        - offspring (from Pygad, crossover of selected parents)
        - change_first: max percentual change of first gene
        - change_rest: max percentual change of the other genes
        - beta_is_gene: if there is an extra parameter in fidelity that
          should be treated as a gene, the mutations applied in said element
          are weaker

    Return:
        - Mutated offspring

    """

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = int(np.random.random() * offspring.shape[1])

        if random_gene_idx == 0:
            offspring[chromosome_idx, random_gene_idx] = offspring[
                chromosome_idx, random_gene_idx
            ] * (1 + random.uniform(-change_first, change_first))
        elif beta_is_gene and random_gene_idx == offspring.shape[1] - 1:
            offspring[chromosome_idx, random_gene_idx] = offspring[
                chromosome_idx, random_gene_idx
            ] * (1 + random.uniform(-0.005, 0.005))
        else:
            variation = random.uniform(-change_rest, change_rest)
            offspring[chromosome_idx, random_gene_idx] = offspring[
                chromosome_idx, random_gene_idx
            ] * (1.0 + variation)

        return offspring


def wmut(offspring, change, beta_is_gene=False):
    """
    Parameters:
        - offsping: (from Pygad, crossover of selected parents)
        - change: Percentual mutation change
        - beta_is_gene: if there is an extra parameter in fidelity that
          should be treated as a gene, the mutations applied are weaker
     Return:
        - offspring: Mutated offspring
    """

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = int(np.random.random() * offspring.shape[1])

        if beta_is_gene and random_gene_idx == offspring.shape[1] - 1:
            offspring[chromosome_idx, random_gene_idx] = offspring[
                chromosome_idx, random_gene_idx
            ] * (1 + random.uniform(-0.005, 0.005))
        else:
            offspring[chromosome_idx, random_gene_idx] = offspring[
                chromosome_idx, random_gene_idx
            ] * (1.0 + random.uniform(-change, change))

    return offspring


def adaptivemut(
    offspring,
    ga_instance,
    weak_change,
    strong_change_first,
    strong_change_rest,
    beta_is_gene=False,
):
    """
    Applies mutation2 until nav instances have passed. Then,
    it checks if the fitness of each of the last nav solutions
    is different (up to some delta) from the mean. If it is, it keeps
    applying mutation2. If it converges (all fitness values are close, up
    to some tolerance, to the mean), it switches to
    wmut (weaker mutation) in order to make a 'finer' tuning of the values.

    Parameters:
        - offspring
        - ga_instance: (defined by Pygad)
        - weak_change: percentual change for wmut
        - strong_change_first: percentual change for first element if
          strong mutation is being applied.
        - strong_change_rest: percentual change for central elements if
          strong mutation is being applied
    Return
        - offspring: mutated offspring
    """
    nav = 10
    tol = 0.001
    history = np.asarray(ga_instance.best_solutions_fitness)
    converge = True

    if history.shape[0] >= nav:
        history = history[-nav:]
        mean = np.mean(history)
        change = abs(history - mean)

        for item in change:
            if item > tol:
                converge = False

        if converge:
            offspring = wmut(offspring, weak_change, beta_is_gene)
        else:
            offspring = mutation2(
                offspring, strong_change_first, strong_change_rest, beta_is_gene
            )

    else:
        offspring = mutation2(
            offspring, strong_change_first, strong_change_rest, beta_is_gene
        )

    return offspring


# Other tools


def closest_odd_integer(number):
    closest_integer = round(number)  # Round the number to the nearest integer

    # if closest_integer <= 1E-4:
    #   return 0

    if closest_integer % 2 != 0:
        return closest_integer

    next_odd = closest_integer + 1
    prev_odd = closest_integer - 1

    if abs(number - prev_odd) < abs(
        number - next_odd
    ):  # Check if the decimal part is closer to the lower integer
        return prev_odd
    else:
        return next_odd


def generate_gsp1(n, maxj, b_is_gene=False):
    """
    Generation of genespace
    Parameters:
        - n = chain length
        - maxj: max. value for couplings
    Return:
        - genespace: of size nj//2
    """
    genespace = []

    nj_genes = (n - 1) // 2 + (1 - n % 2)

    for i in range(0, nj_genes):
        genespace = genespace + [{"low": 0, "high": maxj}]

    if b_is_gene:
        genespace = genespace + [{"low": 0.5, "high": 1, "step": 0.1}]
    return genespace


def generation_func(
    ga,
    n,
    maxj,
    time,
    delta,
    fidelity_tolerance,
    check_tol,
    og_print,
    directory,
    erase_last_gene=False,
    histogram=False,
):
    pop = ga.population

    for ci in range(pop.shape[0]):
        for gi in range(pop.shape[1] - 1 * erase_last_gene):
            while pop[ci, gi] > maxj:
                pop[ci, gi] = pop[ci, gi] * 0.99
            while pop[ci, gi] < 0:
                pop[ci, gi] = pop[ci, gi] + 1.0

        if erase_last_gene:
            while pop[ci, -1] > 1.0:
                pop[ci, -1] = pop[ci, -1] * 0.99
            while pop[ci, -1] < 0.5:
                pop[ci, -1] = pop[ci, -1] + 0.005

    solution, solution_fitness, solution_idx = ga.best_solution()

    fid = fidelity(solution, n, delta, time, erase_last_gene)

    if histogram and (
        ga.generations_completed == 1 or ga.generations_completed % 5 == 0
    ):
        population_histogram(ga, directory)

    if og_print:
        print("Generation", ga.generations_completed)
        print(
            "Solution: ",
            solution,
            "Transmission Fidelity:",
            fid,
            "Fitness: ",
            solution_fitness,
        )
    if check_tol and fid > fidelity_tolerance:
        return "stop"


# hamiltonians and fidelity


def reflect(J, n):
    """
    Creates center-symmetric array from first half of couplings (J),
    for a chain of length n.

    Parameters:
        - J: First half of couplings, including the one corresponding
        to the center of the chain
        - n: length of the chain

    """
    if np.mod(n, 2) == 0:
        nj = n - 1  # number of couplings
        JJ = np.zeros(nj)

        for i in range(0, J.size - 1, 1):  # creates the symmetric chain
            JJ[i] = J[i]
            JJ[nj - i - 1] = J[i]

        JJ[J.size - 1] = J[J.size - 1]

    elif np.mod(n, 2) != 0:
        nj = n - 1  # number of couplings
        JJ = np.zeros(nj)

        for i in range(0, J.size, 1):  # creates the symmetric chain
            JJ[i] = J[i]
            JJ[nj - i - 1] = J[i]

    return JJ


def hxxz(J, n, delta=1.0):
    """
    Constructs XXZ Hamiltonian in the one-excitation basis
    from first half of couplings (full chain is symmetric).
    Default value of the anisotropy parameter (Delta) is set to 1,
    making the default system a Heisenberg Hamiltonian.

    Parameters:
        - J: First half of couplings
        - n: length of the chain
        - delta: anisotropy parameter
    Returns:
        - H: nxn array containing XXZ Hamiltonian
    """

    H = np.full((n, n), 0.0)
    JJ = reflect(J, n)
    sumj = -0.25 * np.sum(JJ)

    for i in range(0, n):
        if i == 0:
            H[i, i] = sumj + 0.5 * JJ[i]
        elif i == n - 1:
            H[i, i] = sumj + 0.5 * JJ[i - 1]
        else:
            H[i, i] = sumj + 0.5 * JJ[i] + 0.5 * JJ[i - 1]

        H[i, i] = H[i, i] * delta

    for i in range(0, n - 1):
        H[i, i + 1] = JJ[i] * (-0.5)
        H[i + 1, i] = H[i, i + 1]

    return H


def diag_hxxz(J, n, delta=1.0):
    """
    Diagonzalizes XXZ Hamiltonian in the one-excitation basis
    from first half of couplings (full chain is symmetric).
    Default value of the anisotropy parameter (Delta) is set to 1,
    making the default system a Heisenberg Hamiltonian.

    Parameters
    ----------
         - J: First half of couplings
         - n: length of the chain
         - delta: anisotropy parameter

    Returns
    -------
        - eigvals: n size array containing XXZ Hamiltonian
        eigenvalues
        - eigvects: nxn size array containing XXZ Hamiltonian
        eigenvectors as columns
    """

    JJ = reflect(J, n)

    sumj = -0.25 * np.sum(JJ)

    d = np.ones(n)
    ds = np.ones(n - 1)

    sumj = -0.25 * np.sum(JJ)

    for i in range(0, n):
        if i == 0:
            d[i] = sumj + 0.5 * JJ[i]
        elif i == n - 1:
            d[i] = sumj + 0.5 * JJ[i - 1]
        else:
            d[i] = sumj + 0.5 * JJ[i] + 0.5 * JJ[i - 1]

        d[i] = d[i] * delta

    for i in range(0, n - 1):
        ds[i] = JJ[i] * (-0.5)

    (eigvals, eigvects) = la.eigh_tridiagonal(d, ds)

    return eigvals, eigvects


#################################################################
#
# fitness functions available
#
#################################################################


def fidelity(J, n, delta=1.0, time=False, erase_last_gene=False):
    """
    Returns transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings. It can also be used as fitness function
    for the genetic algorithm.

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - erase_last_gene = erase last number of input vector J (to be used when
    the provided solutions have been obtained with a genetic algorithm variant
    that stores an extra gene not corresponding to a coupling)

    Returns:
    --------
    -  F = Transmission probability (|<1|n>|²) for the provided system
    """

    if erase_last_gene:
        J = J[:-1]

    (eigvals, eigvects) = diag_hxxz(J, n, delta)

    n = eigvals.size

    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    F = 0.0
    Fr = 0.0
    Fi = 0.0

    for i in range(0, n):
        Fr = Fr + math.cos(eigvals[i] * t) * c1cn[i]
        Fi = Fi + math.sin(eigvals[i] * t) * c1cn[i]

    Fr = np.real(Fr)
    Fi = np.real(Fi)

    F = Fr * Fr + Fi * Fi

    return F


def fidelity_with_eig(J, n, delta=1.0, time=False, erase_last_gene=False):
    """
    Calculates transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings and the associted eigenvalues and eigenvectors.

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - erase_last_gene = erase last number of input vector J (to be used when
    the provided solutions have been obtained with a genetic algorithm variant
    that stores an extra gene not corresponding to a coupling)

    Returns:
    --------
    - F = Transmission probability (|<1|n>|²) for the provided system
    - eigvals = n size array containing XXZ Hamiltonian
    eigenvalues
    - eigvects = nxn size array containing XXZ Hamiltonian
    eigenvectors as columns
    """

    if erase_last_gene:
        J = J[:-1]

    (eigvals, eigvects) = diag_hxxz(J, n, delta)

    n = eigvals.size

    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    F = 0.0
    Fr = 0.0
    Fi = 0.0

    for i in range(0, n):
        Fr = Fr + math.cos(eigvals[i] * t) * c1cn[i]
        Fi = Fi + math.sin(eigvals[i] * t) * c1cn[i]

    Fr = np.real(Fr)
    Fi = np.real(Fi)

    F = Fr * Fr + Fi * Fi

    return F, eigvals, eigvects


def j_fidelity(J, n, delta=1.0, time=False, b_is_gene=False, b=0.9):
    """
    Fitness function based on transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings that also takes into account the "smoothness" of the
    provided solutions. The weight of this property in the final fitness
    of the solution is quantified by beta. Smoothness factor is constructed
    using the square of the difference between consecutive couplings.

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - b_is_gene = the weight value for the "smoothness" of the solution is taken
    as another gene in genetic algorithm
    - b = fix weigth value

    Returns:
    --------
    - F = Solution fitness
    """

    if b_is_gene:
        b = J[-1]
        J = J[:-1]

    (eigvals, eigvects) = diag_hxxz(J, n, delta)

    n = eigvals.size

    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    f = 0.0
    fr = 0.0
    fi = 0.0

    for i in range(0, n):
        fr = fr + math.cos(eigvals[i] * t) * c1cn[i]
        fi = fi + math.sin(eigvals[i] * t) * c1cn[i]

    fr = np.real(fr)
    fi = np.real(fi)

    f = fr * fr + fi * fi

    nj = len(J) #(n - 1) // 2

    ss = 0.0

    for j in range(1, nj):
        ss = ss + (J[j] - J[j - 1]) ** 2

    ss = ss / n**2
    a = 1 - b

    f = f * (a + b * math.exp(-ss))

    return f


def j_mean_fidelity(J, n, delta=1.0, time=False, b_is_gene=False, b=0.9):
    """
    Fitness function based on transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings that also takes into account the "smoothness" of the
    provided solutions. The weight of this property in the final fitness
    of the solution is quantified by beta. Smoothness factor is constructed
    using the difference between consecutive couplings (not squared).

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - b_is_gene = the weight value for the "smoothness" of the solution is taken
    as another gene in genetic algorithm
    - b = fix weigth value

    Returns:
    --------
    - F = Solution fitness
    """

    if b_is_gene:
        b = J[-1]
        J = J[:-1]

    (eigvals, eigvects) = diag_hxxz(J, n, delta)

    n = eigvals.size

    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    f = 0.0
    fr = 0.0
    fi = 0.0

    for i in range(0, n):
        fr = fr + math.cos(eigvals[i] * t) * c1cn[i]
        fi = fi + math.sin(eigvals[i] * t) * c1cn[i]

    fr = np.real(fr)
    fi = np.real(fi)

    f = fr * fr + fi * fi

    nj = (n - 1) // 2

    ss = 0.0

    for j in range(1, nj):
        ss = ss + abs(J[j] - J[j - 1])

    ss = ss / n
    a = 1 - b

    f = f * (a + b * math.exp(-ss))

    return f


def j_meansq_fidelity(J, n, delta=1.0, time=False, b_is_gene=False, b=0.9):
    """
    Fitness function based on transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings that also takes into account the "smoothness" of the
    provided solutions. The weight of this property in the final fitness
    of the solution is quantified by beta. Smoothness factor is constructed
    using the square of the difference of the square of consecutive couplings.

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - b_is_gene = the weight value for the "smoothness" of the solution is taken
    as another gene in genetic algorithm
    - b = fix weigth value

    Returns:
    --------
    - F = Solution fitness
    """

    if b_is_gene:
        b = J[-1]
        J = J[:-1]

    (eigvals, eigvects) = diag_hxxz(J, n, delta)

    n = eigvals.size

    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    f = 0.0
    fr = 0.0
    fi = 0.0

    for i in range(0, n):
        fr = fr + math.cos(eigvals[i] * t) * c1cn[i]
        fi = fi + math.sin(eigvals[i] * t) * c1cn[i]

    fr = np.real(fr)
    fi = np.real(fi)

    f = fr * fr + fi * fi

    nj = (n - 1) // 2

    ss = 0.0

    for j in range(1, nj):
        ss = ss + abs(J[j] ** 2 - J[j - 1] ** 2)

    ss = ss / n**2
    a = 1 - b

    f = f * (a + b * math.exp(-ss))

    return f


def fitness_func_constructor(fid_function, arguments):
    """
    Parameters:
        - fidelity function(can be either fidelity or en_fidelity)
        - arguments: arguments of fidelity functions
    Return:
        - lambda function: the fitness function as required by PyGAD
    """
    fitness = lambda vec: fid_function(vec, *arguments)

    return lambda ga_instance, solution, solution_idx: fitness(solution)


def mutation_func_constructor(mut_function, arguments):
    """
    Parameters:
        - mutation function
        - arguments: arguments of mutation function
    Return:
        - lambda function: the mutation function as required by PyGAD
    """
    mutation = lambda off, ga_instance: mut_function(off, ga_instance, *arguments)

    return lambda offspring, ga_instance: mutation(offspring, ga_instance)


def generation_func_constructor(gen_function, arguments):
    """
    Parameters:
        - mutation function
        - arguments: arguments of mutation function
    Return:
        - lambda function: the mutation function as required by PyGAD
    """

    on_gen = lambda ga_instance: gen_function(ga_instance, *arguments)

    return lambda ga_instance: on_gen(ga_instance)


# file writing


def couplings_to_file(solution, filename, condition):
    """
    Parameters:
        - solution: best solution obtained
        - filename
        - condition: write or append

    Return:
        - saves couplings in file = filename
    """
    with open(filename, condition) as f1:
        writer = csv.writer(f1, delimiter=" ")
        solution = np.asarray(solution)
        for i in range(len(solution)):
            row = ["{:<020}".format(solution[i])]
            writer.writerow(row)

    return True


def fitness_history_to_file(ga_instance, filename):
    """
    Parameters:
        - ga_instance: instance to extract best fitness of each population
        - filename

    Return:
        - saves history in file = filename
    """
    with open(filename, "w") as f2:
        writer = csv.writer(f2, delimiter=" ")
        history = np.asarray(ga_instance.best_solutions_fitness)
        for i in range(len(history)):
            row = ["{:0>3}".format(i), "{:<016}".format(history[i])]
            writer.writerow(row)

    return True


## extra function to use during experiments runs to check
## population fidelity distributions


def population_histogram(ga, directory):
    """
    For a given instance of genetic algorithm, creates a directory
    called hist_frames and plots histograms of population's fidelity
    at the current generation. Is called from inside on_generation
    function in gmod.

    Parameters:
    - ga: genetic algorithm instance (Pygad)
    - directory: to save frames
    Returns:
    - True: if histogram was correctly plotted
    """

    # creates directory if it doesnt exist

    dirname = directory + "/hist_frames"
    isExist = os.path.exists(dirname)
    if not isExist:
        os.mkdir(dirname)

    # access population fitness values and completed generations
    pop_fit = ga.cal_pop_fitness()
    ng = ga.generations_completed

    # plot histogram
    figure, ax = plt.subplots(figsize=(12, 4))
    nbins = 100
    hist, bins, c = ax.hist(
        pop_fit, bins=nbins, range=[0, 1], edgecolor="black", color="#DDFFDD"
    )

    # configure yticks to show percentage of total pop. number
    max_value = int(np.max(hist))
    y = np.linspace(int(0), max_value, 9, dtype=int)
    ax.set_yticks(y)
    ax.set_yticklabels(y * 100 / ga.sol_per_pop)

    x = [0]
    x = x + [i / 10 for i in np.arange(0, 10, 1)]
    ax.set_xticks(x)

    # set grid, title and labels
    plt.grid()
    plt.title("Population distribution for gen. number " + str(ng).zfill(3))
    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Population percentage")

    # save to file
    filename = dirname + "/hist_frame" + str(ng).zfill(3) + ".png"
    plt.savefig(filename)

    return True
