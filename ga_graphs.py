"""
Functions to plot different characteristics and analysis of
solutions for the XXZ hamiltonians obtained using genetic algorithms 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import csv
import pandas as pd
import os
import matplotlib.cm as cm  # matplotlib's color map library
from gmod import *
from mpl_toolkits.mplot3d import Axes3D

# ######################section 1###################################
# Functions to plot attributes of individual solutions such as
# couplings, spectrum and energy differences. These functions take
# coupling values obtained using main programs, generate associated
# Hamiltonians and plot extracted characteristics.
####################################################################


def plot_couplings(files, labels=False, title=False, size=(8, 5), save=False):
    """
    Graphs couplings from solution files.
    Parameters:
    - files: list of coupling files to plot (all in the same graph)
    - labels: list of labels (defaults to file names)
    - title: defaults to 'Resulting couplings'
    - size: image size
    - save: file name to save plot (if no name is provided, plot is not saved)
    """

    files = list(files)

    figure, ax = plt.subplots(figsize=size)

    if not labels:
        labels = files  # sets labels to file names if not provided

    for i in range(len(files)):  # for every file plot J_i vs i
        file = files[i]
        label = labels[i]
        data = np.genfromtxt(file)
        x = np.arange(1, len(data) + 1)
        y = data
        ax.plot(x, y, "o-", label=label)

    ax.set_xlabel("i")
    ax.set_ylabel("J_i")
    ax.legend(loc="lower right")
    ax.set_title("Resulting couplings")
    ax.grid(True)

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(save)

    return True


def plot_normalized_couplings(
    files, labels=False, title=False, size=(8, 5), save=False
):
    """
    Graphs couplings from solution files normalized with chain length.

    Parameters:
    - files: list of coupling files to plot (all in the same graph)
    - labels: list of labels (defaults to file names)
    - title: defaults to 'Resulting couplings'
    - size: image size
    - save: file name to save plot (if no name is provided, plot is not saved)
    """

    figure, ax = plt.subplots(figsize=size)

    if not labels:
        labels = files
    for i in range(len(files)):
        file = files[i]
        label = labels[i]
        data = np.genfromtxt(file)
        nj = len(data) * 2
        x = np.arange(1, len(data) + 1) / nj
        y = data / nj
        ax.plot(x, y, "o-", label=label)

    ax.set_title("Acoplamientos obtenidos")

    if title:
        ax.set_title(title)

    ax.set_xlabel("i/N")
    ax.set_ylabel("J_i")
    ax.legend(loc="lower right")

    if save:
        plt.savefig(save)

    return True


def plot_all_samples(
    directory,
    dimension,
    fid_limit=0.0,
    title=False,
    ns=10,
    transmission_time=False,
    delta=1.0,
    beta_is_gene=False,
):
    """
    Plots every coupling file in a given directory. Calculates fidelity and adds it to
    the label of each plot.
    Parameters:
        - directory: from which to extract the samples
        - dimension: number of couplings of the system which samples are being plotted
        - fid_limit: if a number is specified only samples with fidelity above it are
          plotted. Defaults to 0 (all samples plotted)
        - title: title to add to plot
        - ns: number of samples
        - transmission_time: time to calculate fidelity. If false, uses t = n.
        - delta = anisotropy parameter.
        - beta_is_gene: indicates fidelity function to erase last gene if beta is
          being used as a gene.

    """
    figure, ax = plt.subplots(figsize=(12, 5))

    samples = np.arange(0, ns, 1)

    for sample in samples:
        file = directory + "/jn" + str(dimension) + "sample" + str(sample) + ".dat"
        data = np.genfromtxt(file)
        nj = len(data) * 2
        x = np.arange(1, len(data) + 1)
        y = data
        fid = fidelity(data, delta, transmission_time, beta_is_gene)
        label = "Sample {} with fidelity = {}".format(sample, fid)
        if fid > fid_limit:
            ax.plot(x, y, "o-", label=label)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Solutions obtained in {}".format(directory))

    ax.set_xlabel("i")
    ax.set_ylabel("J_i")
    ax.legend()

    return True


def spectrum(couplings_filename, spectrum_filename, delta=1.0):
    """
    Generates the energy spectrum and saves it in a file from the couplings of
    a XXZ chain.

    Parameters:
        - couplings_filename: file where the couplings are saved.
        - spectrum_filename: file to save energy spectrum
        - delta = value of anisotropy parameter (optional, defaults to 1)
    """
    line = 0

    J = np.genfromtxt(couplings_filename)
    nj = len(J) * 2

    n = nj + 1
    H = np.full((n, n), 0.0)
    JJ = np.zeros(nj)

    for i in range(0, len(J), 1):
        JJ[i] = J[i]
        JJ[nj - i - 1] = J[i]

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

    # Diagonalizamos este hamiltoniano
    (eigvals, eigvects) = la.eig(H)

    eigvals = np.sort(eigvals)

    with open(spectrum_filename, "w") as fw:
        writer = csv.writer(fw, delimiter=" ")

        for j in eigvals:
            writer.writerow(np.array([np.real(j)]))


def gen_enfiles(files, directory="", delta=1.0):
    """
    Generates a list of spectrum files from a list of coupling files
    using spectrum function and saves a file with the spectrum of
    the provided couplings associated hamiltonians.

    Parameters:
        - files: list of coupling files
        - directory: directory name to save spectrum files
        - delta: anisotropy parameter
    Returns:
        - spectrum_files: list with spectrum file names.
    """

    isExist = os.path.exists(directory)  # checks if directory already exists

    if not isExist:  # creates it if it doesnt
        os.mkdir(directory)

    spectrum_files = []  # empty list to save energy filenames

    for file in files:
        spectrum_fname = directory + "/" + file[:-4] + "_spectrum.dat"
        spectrum(file, spectrum_fname, delta)
        spectrum_files.append(spectrum_fname)

    return spectrum_files


def plot_spectrum(files, couplings=True, directory="", delta=1.0):
    """
    Plots energy levels from coupling files, uses spectrum
    to generate the espectra files from couplings. If couplings
    is passed as False it means that the files provided already
    are the energy levels.

    Parameters:
        - files: files to generate and plot energy levels.
        - couplings: If couplings = True, then the files
        provided are couplings associated to a XXZ hamiltonian.
        If false, it means that energy files are provided.
        - directory: directory to save spectrum files if
        couplings is True.
        - delta: anisotropy parameter

    """

    if couplings:
        files = gen_enfiles(files, directory, delta)

    ll = 5  # line length
    nn = (len(files)) // 2

    if len(files) > 2 * nn:
        nn = nn + 1

    fig, axs = plt.subplots(nn, 2, figsize=(10, 10))
    k = 0
    j = 0
    i = 0

    for file in files:
        data = np.genfromtxt(file)
        for value in enumerate(data):
            axs[j, k].plot([0, ll], [value[1], value[1]], "r-")

        # Create a scatter plot
        axs[j, k].set_xlabel("i")
        axs[j, k].set_ylabel("E_i")
        axs[j, k].legend()
        axs[j, k].set_title("spectrum for " + file)
        # plt.show()
        j = j + k
        k = (k + 1) % 2

    fig.tight_layout(h_pad=2, w_pad=2)


def plot_energy_differences(
    files, couplings=True, directory="", delta=1.0, labels=False
):
    """
    Plots normalized energy differences for successive energies
    If couplings is passed as False it means that
    the files provided already are the energy levels.
    Counts how many energy differences are less than
    0.05 apart from the closest odd integer to test
    Kay condition.

    Parameters:
        - files: files to generate and plot energy levels.
        - couplings: If couplings = True, then the files
        provided are couplings associated to a XXZ hamiltonian.
        If false, it means that energy files are provided.
        - directory: directory to save spectrum files if
        couplings is True.
        - delta: anisotropy parameter
        - labels: labels associated to different coupling files
    """
    if couplings:
        files = gen_enfiles(files, directory, delta)

    figure, ax = plt.subplots(figsize=(12, 4))

    if not labels:
        labels = files
    for i in range(len(files)):
        file = files[i]
        label = labels[i]
        data = np.genfromtxt(file)
        data = np.sort(data)
        n = len(data)
        DeltaE = np.empty(n)
        diff = np.empty(n)
        x = np.arange(0, n)
        count = 0
        for i in range(1, n):
            DeltaE[i] = data[i] - data[i - 1]
            # DeltaE_int[i] = closest_odd_integer(DeltaE[i])
            diff[i] = abs(
                DeltaE[i] * (n) / np.pi - closest_odd_integer(DeltaE[i] * n / np.pi)
            )
            if diff[i] < 0.05:
                count += 1

        ax.plot(x, diff, "o-", label=label + "~1 = " + str(count))
        ax.set_xlabel("i")
        ax.set_ylabel("Delta E_i")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 2)
        ax.set_title("Normalized Energy Differences")
    return True


# ###################### section 2 ###################################
# Functions to plot general attributes of solutions such as
# fidelity, CPU time and generations. These functions take
# dataframes summarizing different runs of main programs.
####################################################################

# for contour.py (different chain lengths and delta values,
# one experiment for each)


def contour(df, column="fidelity", title=False):
    """
    Generates contour plot from a dataframe, takes
    dimension and delta (anisotropy parameter) as
    X and Y axis.
    Parameters:
        - df: dataframe to use
        - column: column of df to use as Z axis
        - title: title for the plot
    """

    Z = df.pivot_table(index="dimension", columns="delta", values=column).T.values
    X_unique = np.sort(df.dimension.unique())
    Y_unique = np.sort(df.delta.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if column == "fidelity":
        vmin = 0.9
        vmax = 1
    elif column == "time":
        vmin = 0
        vmax = 2000
    elif column == "generations":
        vmin = 0
        vmax = 5000

    cnt = plt.contourf(X, Y, Z, levels=20, cmap=cm.turbo, vmin=vmin, vmax=vmax)

    if not title:
        title = column + " " + df.name

    ax.set_title(title)
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_xticks(X_unique)

    ax.set_xlabel("Number of couplings")
    ax.set_ylabel("Delta")

    plt.colorbar(label=column)
    plt.grid(True)


def fix_delta_compare(dataframes, delta=1.0):
    """
    For a fixed delta value, plots the different atributes of
    the results (fidelity, cpu time and generations completed)
    for a list of different dataframes

    Parameters:
        - dataframes: list of dataframes to compare
        - delta: anisotropy parameter
    """

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    for dataframe in dataframes:
        ndelta = dataframe[dataframe["delta"] == delta]

        dim = ndelta["dimension"]
        fidelity = ndelta["fidelity"]
        time = ndelta["time"]
        generations = ndelta["generations"]

        # Plot column 1 vs column 2
        axs[0, 0].plot(dim, fidelity, "o-", label=dataframe.name)
        axs[0, 0].set_xlabel("N. of couplings")
        axs[0, 0].set_ylabel("Fidelity")
        axs[0, 0].grid(True)
        # Plot column 1 vs column 3
        axs[0, 1].plot(dim, time, "o-")
        axs[0, 1].set_xlabel("N. of couplings")
        axs[0, 1].set_ylabel("CPU Time")
        axs[0, 1].grid(True)

        # Plot column 1 vs column 4
        axs[1, 0].plot(dim, generations, "o-")
        axs[1, 0].set_xlabel("N. of couplings")
        axs[1, 0].set_ylabel("Generations")
        axs[1, 0].grid(True)

        # Hide the empty subplot
        axs[1, 1].axis("off")

    # Adjust the spacing between subplots
    fig.tight_layout(h_pad=5, w_pad=2)
    fig.legend(loc="center")

    return True


def fix_delta_plot(dataframes, delta, att="fidelity"):
    """
    For a fixed delta value, plots one attribute of
    the results for chains with different number of
    couplings.

    Parameters:
        - dataframes: list of dataframes to compare
        - delta: anisotropy parameter
        - att: attribute to plot (defaults to fidelity)
    """
    axs = plt.figure()

    for dataframe in dataframes:
        ndelta = dataframe[dataframe["delta"] == delta]

        dim = ndelta["dimension"]
        y = ndelta[att]

        plt.plot(dim, y, "o-", label=dataframe.name, linewidth=4)
        plt.xlabel("Dimension")
        plt.ylabel(att)
        plt.title("Delta = " + str(delta))
        plt.tight_layout()
        # Show the plot
    plt.show()
    plt.grid(True)
    plt.legend()


def fix_dim_plot(dataframes, dimension, att="fidelity"):
    """
    For a fixed length of chain value, plots one attribute of
    the results for different values of delta (anisotropy)

    Parameters:
        - dataframes: list of dataframes to compare
        - dimension: number of couplings in the chain
        - att: attribute to plot (defaults to fidelity)
    """
    axs = plt.figure()

    for dataframe in dataframes:
        dim = dataframe[dataframe["dimension"] == dimension]

        delta = dim["delta"]
        y = dim[att]

        plt.plot(delta, y, "o-", label=dataframe.name, linewidth=4)
        plt.xlabel("Delta")
        plt.ylabel(att)

        plt.tight_layout()
        # Show the plot
    plt.show()
    plt.grid(True)
    plt.legend()


# for average.py (statistical analysis)


def compare_mean(
    dataframes,
    title=False,
    attribute="fidelity",
    labels=False,
    figsize=[12, 5],
    save=False,
):
    """
    For a given list of dataframes associated to different runs of genetic algorithm implementation,
    plots average value of some attribute with the associated standard deviation.

    Parameters:
        - dataframes: list of dataframes to analyze
        - title: title for plot
        - attribute: 'fidelity', 'time' (CPU time) or 'generations'
        - labels: labels to identify different dataframes
        - figsize: figure size
        - save: filename to save the plot. If no name is provided, defaults to False
          and does not save the image.

    """

    axs = plt.figure(figsize=figsize)

    if not title:
        title = "Mean " + attribute

    for i in range(len(dataframes)):
        dataframe = dataframes[i]

        if labels:
            label = labels[i]

        grouped_df = dataframe.groupby("dimension")
        mean = grouped_df[attribute].mean()
        std = grouped_df[attribute].std()
        min_value = grouped_df[attribute].min()
        max_value = grouped_df[attribute].max()

        stats = pd.DataFrame(
            {"mean": mean, "std": std, "min": min_value, "max": max_value}
        )

        stats = stats.reset_index()

        plt.errorbar(
            stats["dimension"],
            stats["mean"],
            yerr=stats["std"],
            fmt="o-",
            label=label,
            capsize=4,
            zorder=-2,
        )
    plt.grid(True)
    plt.legend()
    plt.xlabel("N")
    plt.ylabel(attribute)
    plt.title(title)

    if save:
        plt.savefig(save + "mean" + attribute + ".png")
    return True


def compare_maxmin(
    dataframes,
    maxmin="max",
    title=False,
    attribute="fidelity",
    labels=False,
    figsize=[12, 5],
    save=False,
):
    """
    For a given list of dataframes associated to different runs of genetic algorithm implementation,
    plots max or min value of some attribute with the associated standard deviation.

    Parameters:
        - dataframes: list of dataframes to analyze
        - maxmin: plot max or min value of the selected attribute
        - title: title for plot
        - attribute: 'fidelity', 'time' (CPU time) or 'generations'
        - labels: labels to identify different dataframes. If false
            defaults to the list of dataframes
        - figsize: figure size
        - save: filename to save the plot. If no name is provided, defaults to False
          and does not save the image.

    """
    axs = plt.figure(figsize=figsize)

    if not title:
        title = "Mean" + attribute

    for i in range(len(dataframes)):
        dataframe = dataframes[i]

        if labels:
            label = labels[i]

        grouped_df = dataframe.groupby("dimension")
        mean = grouped_df[attribute].mean()
        std = grouped_df[attribute].std()
        min_value = grouped_df[attribute].min()
        max_value = grouped_df[attribute].max()

        stats = pd.DataFrame(
            {"mean": mean, "std": std, "min": min_value, "max": max_value}
        )

        stats = stats.reset_index()
        if maxmin == "max":
            plt.plot(stats["dimension"], stats["max"], "o-", label=label)
        if maxmin == "min":
            plt.plot(stats["dimension"], stats["min"], "o-", label=label)

    plt.grid(True)
    plt.legend()
    plt.xlabel("N")
    plt.ylabel(attribute)
    plt.title(title)

    if save:
        plt.savefig(save + maxmin + attribute + ".png")

    return True


def summary_graph(
    df, title=False, figsize=[15, 5], ylim_min=0.9, ylim_max=1.0, save=False
):
    """
    For a given dataframe plots a 'summary' the associated genetic algorithm run.

    Plot #1: Mean fidelity with standard deviation. Max and min scatter plots with
    color coded points associated to the generations needed to obtain that result.

    Plot #2: Mean, max and min CPU time employed for every dimension.
    Plot #3: Mean, max and min generations employed for every dimension.


    Parameters:
        - dataframe: dataframe to extract data from
        - title (optional): title for plot
        - figsize (optional): figure size
        - ylim_min (optional): lower limit of y range
        - ylim_max (optional): upper limit of y range
        - save: filename to save the plot. If no name is provided, defaults to False
          and does not save the image.

    """

    axs = plt.figure(figsize=figsize)

    grouped_df = df.groupby("dimension")

    mean_fidelity = grouped_df["fidelity"].mean()
    std_fidelity = grouped_df["fidelity"].std()
    min_value_fidelity = grouped_df["fidelity"].min()
    max_value_fidelity = grouped_df["fidelity"].max()

    mean_gen = grouped_df["generations"].mean()
    std_gen = grouped_df["generations"].std()
    min_value_gen = grouped_df["generations"].min()
    max_value_gen = grouped_df["generations"].max()

    mean_time = grouped_df["time"].mean()
    std_time = grouped_df["time"].std()
    min_value_time = grouped_df["time"].min()
    max_value_time = grouped_df["time"].max()

    stats = pd.DataFrame(
        {
            "mean fidelity": mean_fidelity,
            "sd fidelity": std_fidelity,
            "min fidelity": min_value_fidelity,
            "max fidelity": max_value_fidelity,
            "mean generations": mean_gen,
            "sd generations": std_gen,
            "min generations": min_value_gen,
            "max generations": max_value_gen,
            "mean time": mean_time,
            "sd time": std_time,
            "min time": min_value_time,
            "max time": max_value_time,
        }
    )

    stats = stats.reset_index()

    plt.errorbar(
        stats["dimension"],
        stats["mean fidelity"],
        yerr=stats["sd fidelity"],
        fmt="o",
        label="average fidelity",
        capsize=4,
        zorder=-2,
    )

    maxdf = df.groupby("dimension")["fidelity"].max()
    filtered_max = df[df["fidelity"].isin(maxdf)]
    filtered_max = filtered_max.reset_index()

    plt.plot(
        filtered_max["dimension"],
        filtered_max["fidelity"],
        "--",
        label="max fidelity",
        linewidth=1.5,
        color="black",
        zorder=-1,
    )

    sc1 = plt.scatter(
        filtered_max["dimension"],
        filtered_max["fidelity"],
        c=filtered_max["generations"],
        cmap="rainbow",
    )

    for j, txt in enumerate(filtered_max["generations"]):
        plt.annotate(txt, (filtered_max["dimension"][j], filtered_max["fidelity"][j]))

    min_j = df.groupby("dimension")["fidelity"].min()
    filtered_min = df[df["fidelity"].isin(min_j)]
    filtered_min = filtered_min.reset_index()

    plt.plot(
        filtered_min["dimension"],
        filtered_min["fidelity"],
        "--",
        label="min fidelity",
        linewidth=1.5,
        color="grey",
        zorder=-1,
    )

    sc2 = plt.scatter(
        filtered_min["dimension"],
        filtered_min["fidelity"],
        c=filtered_min["generations"],
        cmap="rainbow",
    )

    for i, txt in enumerate(filtered_min["generations"]):
        plt.annotate(txt, (filtered_min["dimension"][i], filtered_min["fidelity"][i]))

    plt.ylim(ylim_min, ylim_max)
    # Add labels and title
    plt.xlabel("N")
    plt.ylabel("Fidelidad")
    if title:
        plt.title(title + " (Fidelity)")

    plt.grid(True)
    plt.legend()

    vmin = min(filtered_max["generations"].min(), filtered_min["generations"].min())
    vmax = max(filtered_max["generations"].max(), filtered_min["generations"].max())
    sc1.set_clim(vmin, vmax)
    sc2.set_clim(vmin, vmax)

    plt.colorbar(label="Generations")

    if save:
        plt.savefig(save + "fid.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].errorbar(
        stats["dimension"],
        stats["mean time"],
        yerr=stats["sd time"],
        fmt="o",
        label="average",
        capsize=4,
    )
    axs[0].plot(
        stats["dimension"], stats["max time"], "--o", label="max", linewidth=2.5
    )
    axs[0].plot(
        stats["dimension"], stats["min time"], "--o", label="min", linewidth=2.5
    )

    # Set the labels and title of the plot
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("CPU time [s]")
    axs[0].set_title("CPU time")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].errorbar(
        stats["dimension"],
        stats["mean generations"],
        yerr=stats["sd generations"],
        fmt="o",
        label="average",
        capsize=4,
    )
    axs[1].plot(
        stats["dimension"], stats["max generations"], "--o", label="max", linewidth=2.5
    )
    axs[1].plot(
        stats["dimension"], stats["min generations"], "--o", label="min", linewidth=2.5
    )

    # Set the labels and title of the plot
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("Gens")
    axs[1].set_title("Generations")
    axs[1].grid(True)

    if save:
        plt.savefig(save + "time_and_gens.png")


def access_best_solutions(
    dataframe, directory, dimensions, return_labels=True, extra_label=" "
):
    """
    For a dataframe containing different runs for different dimensions
    and associated solutions stored in a given directory, returns
    a list of files corresponding to the best solutions for every
    dimension. If labels = True, also returns a list of labels with the
    associated fidelities and dimensions.

    Parameters:
        - dataframe: dataframe to extract data
        - directory: directory where solutions are stored
        - dimensions: dimensions to extract best solutions
        - return_labels: If true, returns list of labels.
        - extra_label: string to add to labels. Defaults to an empty string.
        In that case, label only includes fidelity and dimension.
    Returns:
    - files: list of files with best solutions
    - labels: list of associated labels (if return_labels == true)
    """
    files = []
    labels = []

    for dimension in dimensions:
        ndim = dataframe[dataframe["dimension"] == dimension]
        max_value_index_j_1 = ndim["fidelity"].idxmax() - dimension + 20
        max_value_fidelity_j_1 = ndim["fidelity"].max()
        max_fidelity_solution_j_1 = (
            directory
            + "/jn"
            + str(dimension)
            + "sample"
            + str(max_value_index_j_1)
            + ".dat"
        )

        label = (
            extra_label
            + "fidelity ="
            + str(max_value_fidelity_j_1)
            + ", N = "
            + str(dimension)
        )

        files = files + [max_fidelity_solution_j_1]
        labels = labels + [label]

    if return_labels:
        return files, labels
    else:
        return files
