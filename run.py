import sys
sys.settrace
from scipy.stats import linregress
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_array, check_consistent_length
#from sklearn.metrics._regression import _check_reg_targets
import numpy as np
from scipy.special import xlogy
import warnings
import copy
import os

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """
    Scikit-learn code here ! Code is under BSD 3 clause.
    # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #          Olivier Grisel <olivier.grisel@ensta.org>
    #          Arnaud Joly <a.joly@ulg.ac.be>
    #          Jochen Wersdorfer <jochen@wersdoerfer.de>
    #          Lars Buitinck
    #          Joel Nothman <joel.nothman@gmail.com>
    #          Karan Desai <karandesai281196@gmail.com>
    #          Noel Dawe <noel@dawe.me>
    #          Manoj Kumar <manojkumarsivaraj334@gmail.com>
    #          Michael Eickenberg <michael.eickenberg@gmail.com>
    #          Konstantin Shmelkov <konstantin.shmelkov@polytechnique.edu>
    #          Christian Lorentzen <lorentzen.ch@googlemail.com>
    
      Check that y_true and y_pred belong to the same regression task
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype: str or list, default="numeric"
        the dtype argument passed to check_array
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))
    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'
    return y_type, y_true, y_pred, multioutput

def MAE_for_AML(y_true, y_pred,sample_weight=None,multioutput='uniform_average',**kwargs):
  #Designed to penalize models who passively adapts to median/mean by measuring the slope, then penalizing flat y_pred(y_true) (slope < 0.9)
  #print('Before conversion...')
  y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
  #print('Conversion executed...')
  check_consistent_length(y_true, y_pred, sample_weight)
  #print('Length checked')
  MAE = np.average(np.abs(y_pred - y_true))
  #print('Average computed')
  
  #1st method, suboptimal
  #slope = copy.deepcopy(linregress([x[0] for x in y_true],[x[0] for x in y_pred]).slope)
  #2nd method, buggy
  #y_pred= np.array(y_pred)
  #slope=((y_true*y_pred).mean(axis=1) - y_true.mean()*y_pred.mean(axis=1)) / ((y_true**2).mean() - (y_pred.mean())**2)[0]
  #3rd method
  slope = np.polyfit(y_true.reshape(-1),y_pred.reshape(-1),1)[0]
  
  if np.isnan(slope): slope = 0 
  #print('Slope computed, equals ',slope)
  if slope < 0.9 :
    MAE_more = (1 - slope) * MAE
    #print(int(MAE + MAE_more))
    return MAE + MAE_more
  else : 
    #print('Slope low, score : ',MAE)
    return MAE

def MAE_for_AML_2(y_true, y_pred,sample_weight=None,multioutput='uniform_average',**kwargs):
  #Designed to penalize models who passively adapts to median/mean by measuring the slope, then penalizing flat y_pred(y_true) (slope < 0.9)
  #print('Before conversion...')
  y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
  #print('Conversion executed...')
  check_consistent_length(y_true, y_pred, sample_weight)
  #print('Length checked')
  MAE = np.average(np.abs(y_pred - y_true))
  #print('Average computed')
  
  #1st method, suboptimal
  slope = copy.deepcopy(linregress([x[0] for x in y_true],[x[0] for x in y_pred]).slope)
  #2nd method, buggy
  #y_pred= np.array(y_pred)
  #slope=((y_true*y_pred).mean(axis=1) - y_true.mean()*y_pred.mean(axis=1)) / ((y_true**2).mean() - (y_pred.mean())**2)[0]
  #3rd method
  #slope = np.polyfit(y_true.reshape(-1),y_pred.reshape(-1),1)[0]
  
  if np.isnan(slope): slope = 0 
  #print('Slope computed, equals ',slope)
  if slope < 0.9 :
    MAE_more = (1 - slope) * MAE
    #print(int(MAE + MAE_more))
    return MAE + MAE_more
  else : 
    #print('Slope low, score : ',MAE)
    return MAE
'''
MAE_AML = make_scorer(MAE_for_AML, greater_is_better=False)

from ml_run_lib.age_metaexp import AgeMetaExperiment

nm_agme = AgeMetaExperiment.load_from_pickle('/data/Team_Caspers/Mathieu/data/experiments/exp_age_added_name_1','exp_age_added_name' )

#nm_agme.restart_exp()
'''
print(str(os.path.dirname(os.path.abspath(__file__))))