import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sns

import helper

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

import pickle

import logging

logger = logging.getLogger("newfile.log")

def run():
    try:
        logger.info("Started training the model")
        training_data=pd.read_excel(r"Final Data/final_training_data.xlsx")
        X=training_data.drop(["h1n1_vaccine","seasonal_vaccine"],axis=1)
        y=training_data[["h1n1_vaccine","seasonal_vaccine"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        rf_clf = RandomForestClassifier(n_estimators=300,random_state=0,min_samples_leaf=3,class_weight="balanced").fit(X_train, y_train)
        helper.pickle_dump(rf_clf,"model")
        logger.info("successfully dumped the model to pickle file!")
        y_pred_proba = rf_clf.predict_proba(X_test)   # list of arrays, one per label
        y_score = np.transpose([score[:, 1] for score in y_pred_proba])  # take prob of class=1
        y_true = y_test.to_numpy()

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_thresholds = []

        for i in range(y_true.shape[1]):
            best_f1 = 0
            best_t = 0.5
            for t in thresholds:
                y_pred_bin = (y_score[:, i] >= t).astype(int)   # use new var name
                f1 = f1_score(y_true[:, i], y_pred_bin)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds.append(best_t)

        # Apply thresholds
        y_pred_opt = np.zeros_like(y_score, dtype=int)
        for i, t in enumerate(best_thresholds):
            y_pred_opt[:, i] = (y_score[:, i] >= t).astype(int)

        # ROC AUC
        roc_auc_macro = roc_auc_score(y_true, y_score, average="macro")
        logger.info(f"roc auc score is : {roc_auc_macro:.4f}")

        # Plot ROC curves
        for i, label in enumerate(["h1n1_vaccine", "seasonal_vaccine"]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Multi-Label')
        plt.legend()
        plt.savefig('foo.pdf')

        print("3333333333333333333")
        logger.info("successfully saved the roc_auc curve image!")
        # plt.show()
        

    except Exception as e:
        logger.info(e)

