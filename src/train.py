"""file : train.py
"""
import argparse

import pandas as pd 
import numpy as np 


# use gaussian process as prior
import sklearn.gaussian_process as gp 

# use non linear SVM as machine learning model 
from sklearn.svm import SVC 

# metric for machine learning 
from sklearn.model_selection import cross_val_score

# pickle for model saving
import dill as pickle


parser = argparse.ArgumentParser(description='Train and save a model over the PIMA Indian Diabete')
parser.add_argument('integer', metavar='N', type=int, nargs=1,
                    help='an integer for the grid search')
parser.add_argument('filename', metavar='namefile', type=str, nargs=1,
                    help='filename')
args = parser.parse_args()




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
    CV = cross_val_score(estimator = model, X = X, y = y, scoring = 'roc_auc', cv = 5, n_jobs = -1) 
    print("median CV score for C = {} and gamma = {} : {}".format(10**C, 10**gamma, np.median(CV))) 
    return np.median(CV) 


# grid search over params



def search_best_params(SVC, X, y, n_C = 10, n_gamma = 10):
    params_dict = {"C" : np.linspace(1.2,1.46,n_C), "gamma" : np.linspace(-5.2,-4.9, n_gamma)}
    best_CV = 0
    for C in params_dict["C"]:
        for gamma in params_dict["gamma"]:
            med_cv = cv_score_SVM([C, gamma], SVC, X, y)
            if med_cv > best_CV:
                best_params = [10**C, 10**gamma]
                best_CV = med_cv
    cl = SVC(C = best_params[0], gamma = best_params[1])
    cl.fit(X, y)
    return best_params, cl 


n_C = args.integer[0]
n_gamma = args.integer[0]

name = args.filename[0]

if __name__ == '__main__':
    best_params, final_model = search_best_params(SVC, X, y, n_C, n_gamma)
    with open('../models/'+name,'wb') as nfile:
        pickle.dump(final_model, nfile)


