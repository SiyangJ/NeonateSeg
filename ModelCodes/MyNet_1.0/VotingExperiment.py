import numpy as np
import tensorflow as tf
import sys
import os

sys.argv = ['/usr/bin/python','/proj/NIRAL/users/siyangj/NewModels/model_0227_unet/patch_real_noaug.ini']

import config
from config import FLAGS

from demo import evaluate

prediction_save_dir = FLAGS.prediction_save_dir

STATS_LIST = []
STATS_LIST_FILE = os.path.join(prediction_save_dir,'VotingExperiment.list')

param_range = np.array([15,20,25,30,33,49])

for _i in param_range:
    FLAGS.overlap_add_num = _i
    FLAGS.prediction_save_dir = os.path.join(prediction_save_dir,'overlap_add_num-%d'%_i)
    if not os.path.exists(FLAGS.prediction_save_dir):
        os.mkdir(FLAGS.prediction_save_dir)
    tf.reset_default_graph()
    cur_stats = evaluate()
    with open(STATS_LIST_FILE,'a') as f:
        f.write(str(_i))
        f.write(':  ')
        f.write(np.array2string(np.asarray(cur_stats),separator=', '))
        f.write('\n')
    STATS_LIST += [cur_stats,]

STATS_LIST = np.asarray(STATS_LIST)
best_index = STATS_LIST.mean(axis=1).argmax()
best_param = param_range[best_index]
best_stats = STATS_LIST[best_index,:]

print STATS_LIST
print best_param