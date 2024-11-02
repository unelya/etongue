from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.init_params import N_CLASSES

def accuracy(pred, target):
    target = np.array(target)
    return np.sum(pred == target).astype(np.int32) / target.shape[0]

def calc_metrics(pred, target):
    """
    accuracy: number of right predictions / number of samples
    precision (by class): number of right predictions for class / number of predictions of this class
    recall (by class): number of right predictions for class / number of samples of this class
    confusion matrix: C_{ij} = number of labels that are predicted "i" with true class "j"
    """
    target = np.array(target)
    pred = np.array(pred)
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    fig.tight_layout(pad=3.0)
    bins = range(N_CLASSES+1)
    axs[0].hist(target, bins=bins)
    axs[1].hist(pred, bins=bins)
    axs[0].set_title("Actual")
    axs[1].set_title("Predicted")
    fig.suptitle("Classes Distribution")
    plt.show()
    
    acc = accuracy(pred, target)
    print("accuracy:", acc)
    
    recall = []
    precision = []
    f1 = []
    for label in range(N_CLASSES):
        tp = np.sum(np.logical_and(pred == label, target == label), dtype=np.float32)
        
        r_den = np.sum(target == label, dtype=np.float32)
        r = (tp / r_den) if r_den else 0
        p_den = np.sum(pred == label, dtype=np.float32)
        p = (tp / p_den) if p_den else 0
        
        recall.append(r)
        precision.append(p)
        f1.append(2 * (p * r) / (p + r) if p + r else 0)
        
    classes_names = ['XZ','SZ','DH','MJ','MF']   
    
    fig, axs = plt.subplots(1, 3, figsize=(10,4))
    bins = range(N_CLASSES)
    axs[0].bar(classes_names, precision)
    axs[1].bar(classes_names, recall)
    axs[2].bar(classes_names, f1)
    axs[0].set_title("Precision")
    axs[1].set_title("Recall")
    axs[2].set_title("F1-score")
    plt.show()
    
    classes_names = ['XZ','SZ','DH','MJ','MF']
    matrix = confusion_matrix(target, pred, labels=list(range(N_CLASSES)))
    df_cm = pd.DataFrame(matrix, classes_names, classes_names)
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 16})
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 22})
