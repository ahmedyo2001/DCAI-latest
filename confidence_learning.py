from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from scipy.stats import norm
from sklearn.metrics import accuracy_score


class ConfidenceLearning:
    def __init__(self, model,name, confidence_func: Callable[[Any, np.ndarray], np.ndarray]):
        """
        Constructor for PU learning. The model must be specified. The model should have 'fit'
        method, and 'predict_proba' method.
        param model: Model used for training PU algorithm. The model should have 'fit'
        method, and 'predict_proba' method.
        param hold_out_ratio: ratio specified to hold during training to compute 'c'
        """
        self.name=name
        self.df=pd.DataFrame(columns=["accuracy while poisoned",'Accuracy after heal',"noofchanged",'poisoned_to_neg','poisoned_to_pos','healed to -ve','healed to +ve','changednegtopos',"changedpostoneg"])
        self.model = model
        self.c = None
        self.cneg = None

        self.confidence_func = confidence_func

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function finds all positive labels, holds fraction specified in the constructor, fits
        the remaining features and labels with the model specified in the constructor, and finally
        computes c which is used to find the true probability
        :param x: features. must be np.ndarray
        :param y: noisy labels. must be np.ndarray
        :return:
        """
        
        # Check that x is 2D
        if len(x.shape) != 2:
            raise Exception("Wrong shape for x")

        # Check that y is 1D
        if len(y.shape) != 1:
            raise Exception("Wrong shape for y")

        # Train the model on the entire dataset
        self.model.fit(x, y)

        # Predict probabilities on the same dataset
        hold_prob = self.confidence_func(self.model, x)[:, 1]
        
        # Select only the positive examples
        pos_ind = np.where(y == 1)[0]
        pos_hold_prob = hold_prob[pos_ind]
        
        # Compute c as the mean probability of the positive examples
        # self.c =  np.percentile(pos_hold_prob, 25)
        # Compute c as the truncated mean of the positive examples
        self.c = self.calculate_dynamic_threshold(pos_hold_prob)


        hold_prob = self.confidence_func(self.model, x)[:, 0]
        
        # Select only the positive examples
        pos_ind = np.where(y == 0)[0]
        pos_hold_prob = hold_prob[pos_ind]
        
        # Compute c as the mean probability of the positive examples
        # self.c =  np.percentile(pos_hold_prob, 25)
        # Compute c as the truncated mean of the positive examples
        self.cneg = self.calculate_dynamic_threshold(pos_hold_prob)



    def label_heal(self, x: np.ndarray, y: np.ndarray):
        """
        This function flips negative labels if necessary according to Pu learning. The fit
        function should be called before calling this function to obtain c parameter.
        :param x: features. must be np.ndarray
        :param y: labels including 1 and 0 (only flips 1 if necessary). must be np.ndarray
        :param threshold: threshold for negative label flipping
        :return: new y parameter after updating
        """
        if len(y.shape) == 2:
            y = np.squeeze(y, axis=-1)

        if len(x.shape) != 2:
            raise Exception("Wrong shape for x")

        if len(y.shape) != 1:
            raise Exception("Wrong shape for y")

        neg_ind = np.where(y.astype(int) == 0)[0]

        x_neg = x[neg_ind]
        prob = self.confidence_func(self.model, x_neg)[:, 1]
        #TODO: Remove after testing
        print(prob)
        pred_labels = (prob >= self.c).astype(float)
        y[neg_ind] = pred_labels
        ###################
        neg_ind = np.where(y.astype(int) == 1)[0]

        x_neg = x[neg_ind]
        prob = self.confidence_func(self.model, x_neg)[:, 0]
        print(prob)
        pred_labels = 1-((prob >= self.cneg).astype(float))
        y[neg_ind] = pred_labels

        return pd.Series(y)
    
    def accuracymeasure(self,x_test,y_test):
        y_pred = self.model.predict(x_test)
        acc=accuracy_score(y_test, y_pred)
        return acc
    
    def datarecorder(self,old_y,new_y,poisoned_y,x_test,y_test,noofchanged,poisonedacc):
        y_pred = self.model.predict(x_test)
        acc=accuracy_score(y_test, y_pred)
        pos_to_neg = np.sum((old_y == 1) & (poisoned_y == 0))
        neg_to_pos = np.sum((old_y == 0) & (poisoned_y == 1))
        healedtoneg = np.sum((new_y == 0) & (poisoned_y == 1))
        healedtopos = np.sum((new_y == 1) & (poisoned_y == 0))
        changedpostoneg=np.sum((old_y == 1) & (new_y == 0))
        changednegtopos=np.sum((old_y == 0) & (new_y == 1))
        self.df = pd.concat([self.df,pd.DataFrame({"accuracy while poisoned":poisonedacc,'Accuracy after heal': acc,"noofchanged":noofchanged, 'poisoned_to_neg': pos_to_neg, 'poisoned_to_pos': neg_to_pos,'healed to -ve':healedtoneg,'healed to +ve':healedtopos, 'changednegtopos':changednegtopos, "changedpostoneg":changedpostoneg},index=[self.name])],ignore_index=True)

   

    def calculate_dynamic_threshold(self, pos_hold_prob, percentile=25, trim_percent=0.1):
        trimmed_mean_prob = trim_mean(pos_hold_prob, trim_percent)
        std_dev_prob = np.std(pos_hold_prob)  # You can still use the standard deviation for scaling
        z_score = norm.ppf(percentile / 100)
        return trimmed_mean_prob + z_score * std_dev_prob


def rf_proba(model, X):
    tree_probas = np.array([tree.predict_proba(X) for tree in model.estimators_])
    return tree_probas.mean(axis=0)

def dt_proba(model, X):
    leaf_ids = model.apply(X)
    leaf_probs = model.tree_.value[leaf_ids]
    # Reshape to match number of classes
    num_classes = len(model.classes_)
    leaf_probs = leaf_probs.reshape(-1, num_classes)

    # Add alpha to all counts for Laplace smoothing
    leaf_probs += 0.01
    normalized_probs = leaf_probs / leaf_probs.sum(axis=1, keepdims=True)
    return normalized_probs.squeeze()

def knn_proba(model, X):
    neighbors = model.kneighbors(X, return_distance=False)
    y_train = model._y  # Access the training labels

    probabilities = np.zeros((len(X), len(model.classes_)))  # Initialize empty array
    for i, neighbor_idx in enumerate(neighbors):
        neighbor_labels = y_train[neighbor_idx]
        class_counts = np.bincount(neighbor_labels, minlength=len(model.classes_))
        probabilities[i] = class_counts / class_counts.sum()

    # Assuming binary classification, return the probabilities for the positive class (index 1)
    return probabilities

