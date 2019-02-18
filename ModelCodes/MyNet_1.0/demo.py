import tensorflow as tf
# from model import create_model_hip, create_optimizers,create_model_ivd
from model import create_model_infant_seg, create_optimizers
# from model import create_model_infant_seg_nopooling
from train import train_model
import random
import numpy as np
import os
# from generator import get_training_and_testing_generators
from config import FLAGS
if FLAGS.load_test_with_sitk:
    from predict_multimodality_sitk import  predict_multi_modality_test_images_in_sitk
else:
    from predict_multimodality import  predict_multi_modality_test_images_in_nifti
from copy import deepcopy
from config import FLAGS



def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

def setup_tensorflow():
    
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter('%s' % (FLAGS.checkpoint_dir,), sess.graph_def)

    return sess, summary_writer



class TestData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def test():
    prepare_dirs(delete_train_dir=False)
    sess, summary_writer = setup_tensorflow()
    # here for test, batch_size of tf_input is 1

    (tf_t1_input, tf_t2_input, tf_label, 
            aux1_pred, aux2_pred, main_pred,
            aux1_loss, aux2_loss, main_loss, 
            final_loss, gene_vars, main_possibility) = create_model_infant_seg(train_phase=False)

    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('saver restore from:%s' % model_path)
    saver.restore(sess, model_path)
    
    test_data = TestData(locals())
    if FLAGS.load_test_with_sitk:
        predict_multi_modality_test_images_in_sitk(test_data)
    else:
        predict_multi_modality_test_images_in_nifti(test_data)
    

def main(argv=None):
    print ('>> start testing phase...')
    test()


if __name__ == '__main__':
    tf.app.run()