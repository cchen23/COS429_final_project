# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:03:32 2018

@author: Cathy
"""
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANCE_RATE = 1.0/62
figures_dir = "../figures/results_plots/"

def create_accuracies_plot(accuracies, labels, xlabel, title, savename):
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind, accuracies, width, alpha=0.5)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0,0.5])
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels,fontsize=8)
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.savefig(figures_dir+savename+".png")
    plt.show()
  
def create_all_accuracies(results, labels, xlabel):
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35       # the width of the bars
    num_results = len(results)
    fig, ax = plt.subplots()
    print(len(results))
    print(results)
    for i in range(num_results):
        results_subset = results[i]
        accuracies = results_subset[results_subset['Manipulation Type']=='none']['Test Accuracy']
        ax.bar(ind+(width*i)/num_results, accuracies, width/num_results, alpha=0.5, label=labels[i])

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0,0.5])
    ax.set_title("Test Accuracies")
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabel,fontsize=8)
    ax.set_xlabel("Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir+"default_accuracies_difftraintest"+".png")
    plt.show()

def create_default_accuracies(results, num_train):
    nomanipulation_results = results[results['Manipulation Type']=='none']
    algorithms = ["Chance"] + list(nomanipulation_results['Recognition Algorithm'])
    algorithms = [algorithm.replace(" ","\n") for algorithm in algorithms]
    train_accuracies = [CHANCE_RATE] + list(nomanipulation_results['Train Accuracy'])
    test_accuracies = [CHANCE_RATE] + list(nomanipulation_results['Test Accuracy'])

    create_accuracies_plot(test_accuracies, algorithms, "Algorithm", "Test Accuracies", "default_%d" % num_train)

#def create_manipulation_plot(algorithms, manipulation, results_subset, default_results):
#        num_algorithms = len(algorithms)
#        x_plots = int(num_algorithms/2)
#        y_plots = int(np.ceil(num_algorithms/2))
#        f, axarr = plt.subplots(x_plots,y_plots, sharex=True) # NOTE: HARDCODED
#        for i in range(num_algorithms):
#            algorithm = algorithms[i]
#            algorithm_results = results_subset[results_subset['Recognition Algorithm'] == algorithm]
#            algorithm_accuracies = algorithm_results['Test Accuracy']
#            ind = np.arange(len(algorithm_accuracies))  # the x locations for the groups
#            parameters = list(algorithm_results['Manipulation Parameters'])
#            labels = [list(ast.literal_eval(parameter).values())[0] for parameter in parameters]
#            axarrx = int(i/x_plots)
#            axarry = i%y_plots
#            axarr[axarrx,axarry].bar(ind, algorithm_accuracies)
#            axarr[axarrx,axarry].set_xlabel(algorithm)
#            axarr[axarrx,axarry].set_xticks(ind)
#            axarr[axarrx,axarry].set_xticklabels(labels)
#        plt.title("Test Accuracies with %s" % manipulation.title())
#        plt.savefig(figures_dir+manipulation.replace(" ","")+".png")
#        plt.show()

def create_manipulation_accuracies(results):
    manipulations = np.unique(results['Manipulation Type'])
    algorithms = np.unique(results['Recognition Algorithm'])
    default_results = results[results['Manipulation Type']=='none']
    for manipulation in manipulations:
        if manipulation == "none":
            continue
#        results_subset = results[results['Manipulation Type']==manipulation]
#        results_subset = results_subset.sort_values(by='Manipulation Parameters')
#        create_manipulation_plot(algorithms, manipulation, results_subset, default_results)
        for algorithm in algorithms:
            results_subset = results[results['Manipulation Type']==manipulation][results['Recognition Algorithm']==algorithm]
            accuracies = results_subset['Test Accuracy']
            title=algorithm
            savename="%s_%s"%(manipulation.replace(" ",""),algorithm.replace(" ",""))
            xlabel="Manipulation Parameter"
            parameters = list(results_subset['Manipulation Parameters'])
            labels = [list(ast.literal_eval(parameter).values())[0] for parameter in parameters]
            create_accuracies_plot(accuracies, labels, xlabel, title, savename)

if __name__ == "__main__":
    num_trains = [10, 15, 19]
    results = []
    results.append(pd.read_csv("../results/results.csv", header=0))
    for num_train in num_trains:
        results.append(pd.read_csv("../results/results_%d.csv" % num_train, header=0))
    labels = ["3", "10", "15", "19"]
    xlabel = ["PCA", "Sparse\nRepresentation", "Sparse\nRepresentation\nDimension\nReduction", "Sparse\nRepresentation\nCombined\nL1"]
    create_all_accuracies(results, labels, xlabel)
#    create_default_accuracies(results, num_train)
#    create_manipulation_accuracies(results)