import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import numpy as np
import seaborn as sns
import re

def tablulate_confusion_matrix(classification_report_collection:dict, exclude_models = [], drop_metrics =['macro avg', 'weighted avg'])->pd.DataFrame:
    """
    This is a helper function to tabulate a given set of confusion matrices into a single pandas dataframe. 

    Assumes the input is a dictionary where the keys are the model name, the values are the dictionary returned by classification_report

    Parameters
    ----------
    exclude_models : list[str]
        Which models to exclude from the tabulation. Does a regular expression match so no need to input the entire model name
    drop_metrics : list[str]
        Which metrics to exclude. By default excludes the macro avg and weighted avg
    """
    r = pd.DataFrame()
    for k,v in classification_report_collection.items():
        if any (list(map(lambda x : re.search(x, k) is not None, exclude_models))):
            continue
        tmp = pd.DataFrame(v)
        # Drop desired metrics
        tmp = tmp.drop(drop_metrics, axis=1).T
        # Overwrite incorrect accuracy / support values
        tmp.iloc[2, 0:2] = np.nan
        tmp.iloc[2, 3] = tmp.iloc[0:2, 3].sum()
        tmp = pd.concat({k: tmp}, names=['Model'])
        r = pd.concat([r, tmp])
    # Round of values
    r[r.columns.difference(['suppport'])] = r[r.columns.difference(['suppport'])].round(2)
    r['support'] = r['support'].round(0).astype(int)
    return r

def plot_ROC_PR(y_test, y_probas, model_name, label_to_plot, figsize = (13, 3.5)):
    """
    This is a helper function that plots the ROC AUC & PR curve for a given set of true labels and the 
    probability of an observation falling into that class
    Assumes classification task is binary. Plots the metrics for class label == 1

    Parameters
    ----------
    y_test : The true labels for each observation. 
    y_probas : The computed probability score for an observation being of the class label == 1
    model_name : The model name that generated these results
    label_to_plot : The class label for this classification task

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (13, 3))
    # Fit the ROC curve
    RocCurveDisplay.from_predictions(y_test, y_probas, name = model_name, ax = ax[0])
    ax[0].plot([0, 1], [0, 1], "k--", label="No Skill (AUC = 0.5)") # Random guess benchmark
    ax[0].legend()
    ax[0].set_title(f"ROC for {label_to_plot}")
    # Fit the PR curve
    PrecisionRecallDisplay.from_predictions(y_test, y_probas, name = model_name, ax = ax[1])
    ax[1].plot([0, 1], [0, 0], "k--", label="No Skill")
    # Compute PR statistics
    precision, recall, thresholds = precision_recall_curve(y_test, y_probas)
    # Compute f1-score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    ax[1].scatter(recall[ix], precision[ix], marker='o', color='black',\
                   label=f'Best (Threshold={thresholds[ix]:.2f}, f-score={fscore[ix]:.2f})')
    ax[1].legend()
    ax[1].set_title(f"PR Curve for {label_to_plot}")
    
def plot_precision_recall_vs_threshold(y_test, y_probas, figsize = (13, 3.5)): 
    """
    This is a helper function to plot the precision and recall values against each threshold setting. 
    Assumes classification task is binary. Plots the metrics for class label == 1

    Parameters
    ----------
    y_test : The true labels for each observation. 
    y_probas : The computed probability score for an observation being of the class label == 1

    Returns
    -------
    None
    """
    precisions, recall, thresholds = precision_recall_curve(y_test, y_probas)
    # Compute f1-score
    fscore = (2 * precisions * recall) / (precisions + recall)
    # locate all the indexes of the largest f score. There may be multiple combinations of threshold-precision-recall that produce the greatest fscore
    ixs = np.argwhere(fscore==np.max(fscore))
    # Best fscore and corresponding threshold value
    best_combi = list(map(lambda ix : (thresholds[ix][0], fscore[ix][0], precisions[ix][0], recall[ix][0]) , ixs))
    ## Plotting ##
    fig, ax = plt.subplots(figsize= figsize)
    # Plot the precision
    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.title("Precision, Recall & fscore by threshold value")
    plt.xlabel("Threshold value")
    plt.ylabel("Score")
    # Plot the Recall
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    # Plot the fscore
    plt.plot(thresholds, fscore[:-1], "r-", label="fScore")
    # Plot out all the best fscores
    for best_threshold, best_fscore, best_precision, best_recall in best_combi:
        plt.plot([0, best_threshold], [best_fscore, best_fscore], "k--")
        plt.plot([best_threshold, best_threshold], [0, best_fscore], "k--")
        plt.scatter(best_threshold, best_fscore, marker='o', color='black',\
                    label=f"fscore={best_fscore:.2f} threshold={best_threshold:.2f}\nPrecision={best_precision:.2f} Recall={best_recall:.2f}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    return best_combi

# Helper function to define color palettes
def colors_from_values(values, palette_name):
    values = values.astype(int)
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)