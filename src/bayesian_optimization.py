"""file : bayesian_optimization.py 
   optimizes the hyperparameter of a RBF kernel SVM for Pima Indian Diabetes Dataset
   We use bayesian optimization with Gaussian Process to set the right C and \gamma kernel
   parameters of an SVM model. GP is sued as a surrogate between hyperparameters and cross-validation
   score and Expected Improvement is used as acquisition function
"""

import pandas as pd 
import numpy as np 


# use gaussian process as prior
import sklearn.gaussian_process as gp 

# use non linear SVM as machine learning model 
from sklearn.svm import SVC 

# metric for machine learning 
from sklearn.model_selection import cross_val_score

# start with downloading data
TRAIN = pd.read_csv('../data/training.csv') 
X     = TRAIN.drop(['Outcome'], axis = 1).values
y     = TRAIN.Outcome.values

# start with defining the machine learning model training
def cv_score_SVM(params, SVC, X, y):
    C     = params[0]
    gamma = params[1]

    # Sample C and gamma on the log uniform scale
    model = SVC(C=10**C, gamma=10**gamma)
    return cross_val_score(estimator = model, X = X, y = y, scoring = 'roc_auc', cv = 5, n_jobs = -1)  
