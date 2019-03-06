import numpy as np
import tensorflow as tf
import sys
import os

sys.argv = ['/usr/bin/python','/proj/NIRAL/users/siyangj/NewModels/model_0227_unet/patch_real_error_map_tune.ini']

import config
from config import FLAGS

from model import create_optimizers

from train import train_model
import random
from generator import get_training_and_testing_generators
from copy import deepcopy

if 'bern' in FLAGS.network.lower():
    if FLAGS.stage_1:
        print ">>> **Network**: BernNet Stage 1"
        from BernNet import create_model_infant_seg as create_model
    else:
        print ">>> **Network**: BernNet Stage 2"
        from BernNet import create_model_infant_t1t2dm123_seg as create_model

elif 'unet' in FLAGS.network.lower() or 'u-net' in FLAGS.network.lower():
    if FLAGS.stage_1:
        if 'early' in FLAGS.network.lower():
            print ">>> **Network**: UNet Early Fusion"
            from UNet import create_UNet_early_fusion as create_model
        else:
            print ">>> **Network**: UNet Late Fusion"
            from UNet import create_UNet_late_fusion as create_model
    else:
        print 'Not yet finished'
        sys.exit(0)

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def setup_tensorflow():
    
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    tf.gfile.MkDir('%s/training_log' % (FLAGS.checkpoint_dir,))
    tf.gfile.MkDir('%s/validation_log' % (FLAGS.checkpoint_dir,))
    summary_writer = tf.summary.FileWriter('%s/training_log' % (FLAGS.checkpoint_dir,), sess.graph)
    val_sum_writer = tf.summary.FileWriter('%s/validation_log' % (FLAGS.checkpoint_dir,), sess.graph)
    
    if FLAGS.show_test_in_training:
        tf.gfile.MkDir('%s/test_log' % (FLAGS.checkpoint_dir,))
        test_sum_writer = tf.summary.FileWriter('%s/test_log' % (FLAGS.checkpoint_dir,), sess.graph)
        return sess, summary_writer, val_sum_writer, test_sum_writer

    return sess, summary_writer, val_sum_writer

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

print '>>> STAGE %d TRAINING <<<' % (1 if FLAGS.stage_1 else 2)

model_ret = create_model(train_phase=True)

(tf_t1_input, tf_t2_input, tf_label, 
        aux1_pred, aux2_pred, main_pred,
        aux1_loss, aux2_loss, main_loss, 
        final_loss, gene_vars, main_possibility) = model_ret[:12]
if not FLAGS.stage_1:
    tf_dm_input1, tf_dm_input2, tf_dm_input3 = model_ret[12:15]
if FLAGS.use_error_map:
    tf_weight_main = model_ret[-1]

print '>>> MODEL CREATED'
zero_ops, accum_ops, train_minimize, learning_rate, global_step = create_optimizers(final_loss)
print '>>> OPTIMIZER CREATED'

train_data = TrainData(locals())

from predict_multimodality_sitk import regenerate_error_map

checkpoint_dir = FLAGS.checkpoint_dir
STATS_LIST = []

param_name = 'error_map_correct_weight'
param_range = np.linspace(0.1,0.9,9)

assert param_range is not None, 'Not implemented yet!'

if param_range is not None:
    for param in param_range:

        setattr(FLAGS,param_name,param)
        regenerate_error_map(prediction_save_dir='/proj/NIRAL/users/siyangj/NewModels/model_0227_unet/models/1/patch_pretrain1/')
        FLAGS.checkpoint_dir = os.path.join(checkpoint_dir,'%s-%s'%(param_name,param))
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.mkdir(FLAGS.checkpoint_dir)
        prepare_dirs(delete_train_dir=False)

        if FLAGS.show_test_in_training:
            sess, summary_writer, val_sum_writer, test_sum_writer = setup_tensorflow()
        else:
            sess, summary_writer, val_sum_writer = setup_tensorflow()
            
        train_data.__dict__.update(locals())
        print '>>> TRAINING START'
        cur_stats = train_model(train_data)
        with open(os.path.join(checkpoint_dir,'CorrectWeightExperiment.list'),'a') as f:
            f.write(str(param))
            f.write(':  ')
            f.write(np.array2string(np.asarray(cur_stats),separator=', '))
            f.write('\n')
        STATS_LIST += [cur_stats,]

    STATS_LIST = np.asarray(STATS_LIST)
    best_index = STATS_LIST.mean(axis=1).argmax()
    best_param = param_range[best_index]
    best_stats = STATS_LIST[best_index,:]
else:
    pass

print STATS_LIST,best_param

with open(os.path.join(checkpoint_dir,'CorrectWeightExperiment.list'),'a') as f:
    f.write('>>> Best param:  %f\n'%best_param)
    f.write('>>> Best stats:  ')
    f.write(np.array2string(np.asarray(best_stats),separator=', '))
    f.write('\n')

os.system('cp -r %s %s'%(os.path.join(checkpoint_dir,'%s-%s'%(param_name,best_param),'*'),checkpoint_dir))