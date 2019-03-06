import os

import tensorflow as tf
import numpy as np
from config import FLAGS
import main

## Tuning one parameter
#  Start with certain parameter values
#  With certain directories
#  Get results with main.train()
#  Store the results somewhere in cache
#  Reset the parameter values
#  Reset the directories
#  Redo main.train()
#  ...
#  Compare the results
#  Based on some criterion
#  Select the best one or the best several
def tune_one_parameter(param_name,param_range=None,param_max=None,param_min=None,max_tune_times=None):
    
    checkpoint_dir = FLAGS.checkpoint_dir
    STATS_LIST = []
    
    assert param_range is not None, 'Not implemented yet!'
    
    if param_range is not None:
        for param in param_range:

            setattr(FLAGS,param_name,param)
            FLAGS.checkpoint_dir = os.path.join(checkpoint_dir,'%s-%s'%(param_name,param))
            os.mkdir(FLAGS.checkpoint_dir)
            STATS_LIST += [main.train(),]

        STATS_LIST = np.asarray(STATS_LIST)
        best_index = STATS.mean(axis=1).argmax()
        best_param = param_range(best_param)
    else:
        pass
    
    os.system('cp %s %s'%(os.path.join(checkpoint_dir,'%s-%s'%(param_name,best_param),'*'),checkpoint_dir))
    
    return STATS_LIST, best_param


def tune_multiple_parameters(param_names,param_ranges,tune_iterations=1):
    STATS_LISTS = []
    best_params = []
    for tune_iter in xrange(tune_iterations):
        for _i,(param_name,param_range) in enumerate(zip(param_names,param_ranges)):
            STATS_LIST, best_param = tune_one_parameter(param_name,param_range)
            setattr(FLAGS,param_name,best_param)
            if tune_iter == 0:
                STATS_LISTS += [STATS_LIST,]
                best_params += [best_param,]
            else:
                STATS_LISTS[_i] = STATS_LIST
                best_params[_i] = best_param
    return STATS_LISTS, best_params
