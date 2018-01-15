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
  
def create_manipulation_accuracies(num_train):
    manipulation_parameter_names = {
            'occlude_lfw':'Occlusion Window Size',
            'radial_distortion':'k',
            'blur':'Blur Window Size',
            'dfi':'',
            }
    results = pd.read_csv("../results/results_%d.csv" % num_train, header=0)
    manipulations = np.unique(results['Manipulation Type'])
    #manipulations = ['dfi']
    algorithms = list(np.unique(results['Recognition Algorithm'])) # FOR DFI
    #algorithms.remove('VGG') #FOR DFI
    for manipulation in manipulations:
        if manipulation == "none" or manipulation == 'dfi': # TODO: TAKE OUT DFI PART ONCE WE HAVE RESULTS.
            continue
        manipulation_results = results[results['Manipulation Type']==manipulation]
        parameters = list(set(manipulation_results['Manipulation Parameters']))
        parameters.sort()
        parameter_labels = [list(ast.literal_eval(parameter).values())[0] for parameter in parameters]
        
        ind = np.arange(len(algorithms))  # the x locations for the groups
        width = 0.35       # the width of the bars
        num_results = len(parameter_labels)
        fig, ax = plt.subplots()
        
        for i in range(len(parameters)):
            parameter_results = manipulation_results[manipulation_results['Manipulation Parameters']==parameters[i]].sort_values('Recognition Algorithm')
            accuracies = parameter_results['Test Accuracy']
            ax.bar(ind+(width*(i-num_results/2))/num_results, accuracies, width/num_results, alpha=0.5, label=parameter_labels[i])
        ax.set_xlabel("Recognition Algorithm")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Accuracies with %s Manipulation Images" % manipulation.title())
        ax.set_xticks(ind)
        algorithms_labels = [algorithm.replace(" ","\n") for algorithm in algorithms]
        ax.set_xticklabels(algorithms_labels,fontsize=8)
        ax.set_xlabel("Algorithm")
        plt.legend(title=manipulation_parameter_names[manipulation])
        plt.tight_layout()
        plt.savefig(figures_dir+"results_%s_%d.png"%(manipulation.replace(" ",""), num_train))
        plt.show()

def create_all_traintestsplit_default_accuracies():
    num_trains = [3, 10, 15, 19]
    results = []
    for num_train in num_trains:
        results.append(pd.read_csv("../results/results_%d.csv" % num_train, header=0))
    labels = ["3", "10", "15", "19"]
    xlabel = ["PCA", "Sparse\nRepresentation", "Sparse\nRepresentation\nDimension\nReduction", "Sparse\nRepresentation\nCombined\nL1", "SVM", "VGG"]

    ind = np.arange(len(xlabel))  # the x locations for the groups
    width = 0.35       # the width of the bars
    num_results = len(results)
    fig, ax = plt.subplots()
    for i in range(num_results):
        results_subset = results[i]

        accuracies = results_subset[results_subset['Manipulation Type']=='none']['Test Accuracy']
        ax.bar(ind+(width*(i-num_results/2))/num_results, accuracies, width/num_results, alpha=0.5, label=labels[i])

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0,1.0])
    ax.set_title("Test Accuracies")
    ax.set_xticks(np.arange(len(xlabel))) # ind
    ax.set_xticklabels(xlabel,fontsize=8)
    ax.set_xlabel("Algorithm")
    plt.legend(title="Training Faces Per Person")
    plt.tight_layout()
    plt.savefig(figures_dir+"default_accuracies_difftraintest"+".png")
    plt.show()

#def create_default_accuracies(results, num_train):
#    nomanipulation_results = results[results['Manipulation Type']=='none']
#    algorithms = ["Chance"] + list(nomanipulation_results['Recognition Algorithm'])
#    algorithms = [algorithm.replace(" ","\n") for algorithm in algorithms]
#    train_accuracies = [CHANCE_RATE] + list(nomanipulation_results['Train Accuracy'])
#    test_accuracies = [CHANCE_RATE] + list(nomanipulation_results['Test Accuracy'])
#
#    create_accuracies_plot(test_accuracies, algorithms, "Algorithm", "Test Accuracies", "default_%d" % num_train)

if __name__ == "__main__":
    create_all_traintestsplit_default_accuracies()
    #create_manipulation_accuracies(15)
