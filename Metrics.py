import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

class Metric:
    def print_classification_metrics(y_test, y_pred):
        params = pd.DataFrame()
        params.index = ['Accuracy', 'Precision', 'Recall', 'F1 score']
        for label in pd.DataFrame(y_test)[0].unique():
            TP, TN, FP, FN = Metric.__get_classifiers(y_test, y_pred, label)
            params[label] = [Metric.__get_accuracy(TP, TN, FP, FN), Metric.__get_precision(TP, FP), Metric.__get_recall(TP, FN), Metric.__get_f1(TP, FP, FN)]
        print(params)
    def __get_accuracy(TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)
    def __get_precision(TP, FP):
        return TP / (TP + FP)
    def __get_recall(TP, FN):
        return TP / (TP + FN)
    def __get_f1(TP, FP, FN):
        return 2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))
    def __get_classifiers(y_test, y_pred, label):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y_test)):
            if y_test[i] == label:
                if y_test[i] == y_pred[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if y_test[i] == y_pred[i]:
                    TN += 1
                else: 
                    FP += 1
        return (TP, TN, FP, FN)

class Plot:
    def plot_confusion_matrix(y_test, y_pred):
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)
    def plot_roc_curve(y_test, y_pred, labels):
        f, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], linestyle="--")
        for label in labels:
            RocCurveDisplay.from_predictions(y_test, y_pred, pos_label=label, ax=ax)
    def plot_history_trend(history, metric):
        f, ax = plt.subplots()
        plt.plot(range(len(history)), history, label=metric)
        ax.set_title(metric)
        ax.set_xlabel("Эпоха")
        ax.legend(loc='upper left')
        plt.show()