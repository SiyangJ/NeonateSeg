import tensorflow as tf
from model import create_optimizers

from train import train_model
import random
import numpy as np
import os
import sys
from generator import get_training_and_testing_generators
from copy import deepcopy
from config import FLAGS

if 'bern' in FLAGS.network.lower():
    if FLAGS.stage_1:
        print ">>> **Network**: BernNet Stage 1"
        from BernNet import create_model_infant_seg as create_model
    else:
        print ">>> **Network**: BernNet Stage 2"
        from BernNet import create_model_infant_t1t2dm123_seg as create_model

elif 'unet' in FLAGS.network.lower() or 'u-net' in FLAGS.network.lower():
    if 'early' in FLAGS.network.lower():
        print ">>> **Network**: UNet Early Fusion"
        from UNet import create_UNet_early_fusion as create_model
    else:
        print ">>> **Network**: UNet Late Fusion"
        from UNet import create_UNet_late_fusion as create_model
    '''
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
    '''

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

    ## Editted by Siyang Jing on Nov 4
    ## Try to add validation summary writer
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


def train():
    
    print 'main: stage_1 = %s'%FLAGS.stage_1
    print '>>> STAGE %d TRAINING <<<' % (1 if FLAGS.stage_1 else 2)
    
    prepare_dirs(delete_train_dir=False)

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
    
    if FLAGS.show_test_in_training:
        sess, summary_writer, val_sum_writer, test_sum_writer = setup_tensorflow()
    else:
        sess, summary_writer, val_sum_writer = setup_tensorflow()
    
    train_data = TrainData(locals())
    print '>>> TRAINING START'
    
    train_model(train_data)
    
    
def __test():
    
    print '>>> STAGE %d DEBUGGING <<<' % (1 if FLAGS.stage_1 else 2)
    
    prepare_dirs(delete_train_dir=False)

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
    
    if FLAGS.show_test_in_training:
        sess, summary_writer, val_sum_writer, test_sum_writer = setup_tensorflow()
    else:
        sess, summary_writer, val_sum_writer = setup_tensorflow()
    
    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('saver restore from:%s' % model_path)
    saver.restore(sess, model_path)
    
    train_data = TrainData(locals())
    return train_data


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()