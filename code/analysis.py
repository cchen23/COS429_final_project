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
    for manipulation in manipulations:
        if manipulation == "none":
            continue
        manipulation_results = results[results['Manipulation Type']==manipulation]
        algorithms = list(np.unique(manipulation_results['Recognition Algorithm']))
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
        ax.set_ylim(0, 1)
        algorithms_labels = [algorithm.replace(" ","\n") for algorithm in algorithms]
        ax.set_xticklabels(algorithms_labels,fontsize=8)
        ax.set_xlabel("Algorithm")
        plt.legend(title=manipulation_parameter_names[manipulation])
        plt.tight_layout()
        plt.savefig(figures_dir+"results_%s_%d.png"%(manipulation.replace(" ",""), num_train))

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

def compute_manipulation_impact(num_train):
    results = pd.read_csv("../results/results_%d.csv" % num_train, header=0)
    manipulations = list(np.unique(results['Manipulation Type']))
    algorithms = list(np.unique(results['Recognition Algorithm']))
    columns = ["Algorithm", "default"]
    columns += [manipulation+"_normalizeddifference" for manipulation in manipulations]
    columns += [manipulation+"_difference" for manipulation in manipulations]
    #    columns += [manipulation+"all" for manipulation in manipulations]
    manipulation_impact_info = pd.DataFrame(index=algorithms,columns=columns)
    manipulation_impact_info["Algorithm"] = algorithms
    for algorithm in algorithms:
        algorithm_results = results[results['Recognition Algorithm'] == algorithm]
        default_accuracy = algorithm_results[algorithm_results['Manipulation Type']=='none']['Test Accuracy'].values[0] 
        manipulation_impact_info.loc[manipulation_impact_info.Algorithm == algorithm, 'default'] = default_accuracy
        for manipulation in manipulations:
            manipulation_accuracies = algorithm_results[algorithm_results['Manipulation Type']==manipulation].sort_values('Manipulation Parameters')['Test Accuracy']

            mean_accuracy = np.mean(manipulation_accuracies.values)
            manipulation_impact_info.loc[manipulation_impact_info.Algorithm == algorithm, manipulation+"_normalizeddifference"] = (default_accuracy - mean_accuracy) / default_accuracy
            manipulation_impact_info.loc[manipulation_impact_info.Algorithm == algorithm, manipulation+"_difference"] = default_accuracy - mean_accuracy
#            manipulation_impact_info[algorithm, manipulation+"all"] = " ".join([str(accuracy) for accuracy in manipulation_accuracies])
    manipulation_impact_info.to_csv('../results/manipulation_impact.csv')

def plot_manipulation_impact(normalized_average=True):
    keyword = "_normalizeddifference" if normalized_average else "_difference"
    manipulation_labels = {
            'occlude_lfw%s' % keyword:'Occlusion',
            'radial_distortion%s' % keyword:'Radial Distortion',
            'blur%s' % keyword:'Blur',
            }
    manipulation_impact_info = pd.read_csv('../results/manipulation_impact.csv', header=0)
    manipulation_impact_info = manipulation_impact_info.set_index('Algorithm')
    manipulations = [manipulation for manipulation in list(manipulation_impact_info.columns) if keyword in manipulation]
    manipulations.remove('dfi%s' % keyword)
    manipulations.remove('none%s' % keyword)
    algorithms = list(manipulation_impact_info.index)
    ind = np.arange(len(algorithms))  # the x locations for the groups
    width = 0.35       # the width of the bars
    num_results = len(manipulations)+1
    fig, ax = plt.subplots()
    
    ax.bar(ind-width*(num_results/2)/num_results, manipulation_impact_info['default'].values, width/num_results, alpha=0.5, label="Default Accuracy", color="grey")
    for i in range(len(manipulations)):
        values = manipulation_impact_info[manipulations[i]]
        ax.bar(ind+(width*(i+1-num_results/2))/num_results, values, width/num_results, alpha=0.5, label=manipulation_labels[manipulations[i]])
    ax.set_ylabel("Difference")
    ax.set_title("Difference Between Accuracy on Default and Manipulated Images")
    ax.set_xticks(ind)
    algorithms_labels = [algorithm.replace(" ","\n") for algorithm in algorithms]
    ax.set_xticklabels(algorithms_labels,fontsize=8)
    ax.set_xlabel("Algorithm")
    lgd = plt.legend(title="Manipulation Type",bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(figures_dir+"manipulation_impact%s.png" % keyword,bbox_extra_artists=(lgd,), bbox_inches='tight')
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
    # create_all_traintestsplit_default_accuracies()
    # create_manipulation_accuracies(15)
    compute_manipulation_impact(15)
    plot_manipulation_impact(True)
    plot_manipulation_impact(False)