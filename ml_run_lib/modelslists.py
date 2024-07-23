# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:09:38 2018

@author: Mathieu
"""

# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import sklearn
from sklearn.metrics import cohen_kappa_score, make_scorer


# Check the TPOT documentation for information on the structure of config dicts

classifier_dict = {
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 12,5),
        'min_samples_split': range(2, 22,5),
        'min_samples_leaf': range(1, 22,5)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.20),
        'min_samples_split': range(2, 22,5),
        'min_samples_leaf': range(1, 22,5),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.20),
        'min_samples_split': range(2, 22,8),
        'min_samples_leaf':  range(1, 22,8),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1,12,5),
        'min_samples_split': range(2,12,5),
        'min_samples_leaf': range(1,12,5),
        'subsample': np.arange(0.05, 1.01, 0.2),
        'max_features': np.arange(0.05, 1.01, 0.2)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1,200,10),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },
    
    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 12,5),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 22,5),
        'nthread': [1]
    },
    
    'sklearn.neighbors.NearestCentroid':{
        'shrink_threshold':[None,0.2,0.4,0.6,0.8]
    },
    
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis':{
        'solver':['lsqr','eigen'],
        'shrinkage':[None,'auto',0.2,0.4,0.6,0.8],
        'tol':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },
    
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis':{
        'reg_param':[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.2, 0.5, 0.8 ],
        'tol':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
    
    'sklearn.linear_model.LogisticRegressionCV':{
        'penalty':['l2'],
        'Cs':[1, 5, 10, 15, 20, 25],
        'fit_intercept':[True],
        'dual':[False],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'scoring':[make_scorer(cohen_kappa_score)],
        'cv':[3]
    },
#    'false.entry.fortest': {
#            }
    }
    
classifier_dict_2={
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis':{
        'solver':['svd','lsqr','eigen'],
        'tol':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
    'sklearn.linear_model.LogisticRegressionCV':{
        'penalty':['l2'],
        'Cs':[1, 5, 10, 15, 20, 25],
        'fit_intercept':[True],
        'dual':[False],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'scoring':[make_scorer(cohen_kappa_score)]
    }
    }
    
test = {
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]},
    
                                                         
     'sklearn.ensemble.AdaBoostClassifier': {
             'base_estimator': [{
                     'sklearn.naive_bayes.GaussianNB': {
                             'priors': [None]}},
                    'gini', 'entropy'],
             'base_estimator__splitter' :   ['best', 'random'],
             'n_estimators': [1, 2]            
                     
                     
                     }
     


    
    
#    'sklearn.svm.LinearSVC': {
#        'penalty': ["l1","l2"],
#        'loss': ["squared_hinge"],
#        'dual': [True],
#        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
#    }
    
        }

    # Preprocesssors

preprocessor_dict = {
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 12,5)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 12,5)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100,5),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}
            
complete_dict={**classifier_dict, **preprocessor_dict}