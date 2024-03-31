

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


from utils.load_data import load_data_train, load_data_test, load_data_val

from sklearn.metrics import  accuracy_score,f1_score,balanced_accuracy_score


import matplotlib.lines as mlines


X_train,y_train = load_data_train()
X_test,y_test = load_data_test()


def take_subset(i):
    models_folder = "trained_models"

# loop through each subset folder

    subset_folder = f"{models_folder}/subset_{i}"

    # get the list of models in the subset folder
    model_files = [f for f in os.listdir(subset_folder) if f.endswith('.joblib')]

    colors = ['lightcoral', 'sandybrown', 'khaki', 'violet', 'olive', 'springgreen','brown','silver','cyan']

    n_plots = len(model_files)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    

    # loop through each model file in the subset folder
    for j, model_file in enumerate(model_files):
        # load the model from the joblib file
        model = joblib.load(f"{subset_folder}/{model_file}")
        model_pred = model.predict(X_test)
        # print(f"f1score:{f1_score(model_pred,y_test)}")
       

        model_name = model_file.split("-")[0].strip()
        f1 = f1_score(model_pred, y_test)
        accuracy = balanced_accuracy_score(model_pred, y_test)
        print(f"f1 score for {model_name}: {f1}")
        print(f"accuracy for {model_name}: {accuracy}")
        
        if i >= len(axes):
            break
        ax = axes[j]
        ax.set_title(f"{model_name}",fontsize=20)
        color = colors[model_files.index(model_file) % len(colors)]
        ax.plot(np.arange(len(model_pred[19000:23000])), model_pred[19000:23000], color=color)
       
        
        ax.set_xlabel(("Timestamp"),fontsize=17)
        ax.set_ylabel(("Prediction"),fontsize=17)
        ax.set_yticks([0.0, 1.0])
        
        

    for ax in axes[n_plots:]:
        
        ax.axis('off')

    last_ax = fig.add_subplot(n_rows, n_cols, n_plots+1)
    last_ax.set_title(('Test Data'),fontsize=20)
    last_ax.plot(np.arange(len(y_test[19000:23000])), y_test[19000:23000], color='red') 
    last_ax.set_xlabel(("Timestamp"),fontsize=17)
    last_ax.set_ylabel(("Prediction"),fontsize=17)
    fig.suptitle((f"Subset {i+1}"),fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # add legend
   # create custom legend handles
    normal_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='0-Normal data')
    attack_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='1-Attack data')

    # add legend with custom handles
    fig.legend(handles=[normal_handle, attack_handle], loc='lower center', ncol=2,fontsize=18)

   

    plt.show()
   

    # save the plot
    plot_filename = f"plot_subset_{i}.png"
    fig.savefig(plot_filename)


def plot_model_for_subset(i):
    models_folder = "trained_models"

    # loop through each subset folder
    subset_folder = f"{models_folder}/subset_{i}"

    # get the list of models in the subset folder
    model_files = [f for f in os.listdir(subset_folder) if f.endswith('.joblib')]

    colors = ['lightcoral', 'sandybrown', 'khaki', 'violet', 'olive', 'springgreen','brown','silver','cyan']

    f1_scores = []
    accuracies = []

    # loop through each model file in the subset folder
    for j, model_file in enumerate(model_files):
        # load the model from the joblib file
        model = joblib.load(f"{subset_folder}/{model_file}")
        
        # make predictions on the test data
        y_pred = model.predict(X_test)

        # calculate the F1 score and accuracy of the model
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # add the F1 score and accuracy to the lists
        f1_scores.append(f1)
        accuracies.append(accuracy)

    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,10))

    # plot the F1 scores
    for j, f1 in enumerate(f1_scores):
        ax1.vlines(j, ymin=0, ymax=f1, color=colors[j], linewidth=10)
    ax1.set_title("F1 Scores")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(f"Models trained on Subset {i+1}")
    ax1.set_xticks([])
    

    # plot the accuracies
    for j, accuracy in enumerate(accuracies):
        ax2.vlines(j, ymin=0, ymax=accuracy, color=colors[j], linewidth=10)
    ax2.set_title("Accuracies")
    ax2.set_xlabel(f"Models trained on Subset {i+1}")
    ax2.set_xticks([])
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    

    # add a legend for the colors
    # add a legend for the colors
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(model_files))]
    labels = [model_file.split("-")[0].strip() for model_file in model_files]
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 1.02),fontsize=20)


    

    # adjust the layout of the plots
    plt.subplots_adjust(hspace=0.5)

    # show the plots
    plt.show()


        
   