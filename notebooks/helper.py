import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score

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
    ax[1].plot([0, 1], [0, 0], "k--", label="No Skil")
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
    fig = plt.figure(figsize= figsize)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.title("Precision & Recall by threshold value")
    plt.xlabel("Threshold value")
    plt.ylabel("Score")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.legend()