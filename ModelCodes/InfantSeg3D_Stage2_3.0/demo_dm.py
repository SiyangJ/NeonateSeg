import tensorflow as tf
from model import create_model_infant_t1t2dm123_seg
from train import train_model
import random
import numpy as np
import os
from predict_dm import  predict_multi_modality_test_images_in_nifti
from copy import deepcopy
from config import FLAGS



def prepare_dirs(delete_checkpoint_dir=False):
	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
	
	# Cleanup train dir
	if delete_checkpoint_dir:
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

	summary_writer = tf.summary.FileWriter('%s' % (FLAGS.checkpoint_dir,), sess.graph_def)

	return sess, summary_writer



class TestData(object):
	def __init__(self, dictionary):
		self.__dict__.update(dictionary)




def test():
	prepare_dirs(delete_checkpoint_dir=False)
	sess, summary_writer = setup_tensorflow()


	(tf_t1_input, tf_t2_input, tf_dm_input1, tf_dm_input2, tf_dm_input3, tf_label, 
            aux1_pred, aux2_pred, main_pred,
            aux1_loss, aux2_loss, main_loss, 
            final_loss, gene_vars, main_possibility) = create_model_infant_t1t2dm123_seg(train_phase=False)

	saver = tf.train.Saver()
	model_path = tf.train.latest_checkpoint(FLAGS.last_trained_checkpoint)
	print('saver restore from:%s' % model_path)
	saver.restore(sess, model_path)
	

	test_data = TestData(locals())

	predict_multi_modality_test_images_in_nifti(test_data)
	

	

def main(argv=None):
	print ('>> start testing phase...')
	test()




if __name__ == '__main__':
	tf.app.run()