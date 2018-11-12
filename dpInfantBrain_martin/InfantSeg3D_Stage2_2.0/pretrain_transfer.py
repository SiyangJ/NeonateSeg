import tensorflow as tf
import numpy as np
from config import FLAGS

import h5py
import sys

hip_3d_hdf5_data = None
sports_3d_hdf5_data = None


def open_hdf5_file():
	global hdf5_data
	# if FLAGS.with_context:
	# 	hdf5_data = h5py.File(FLAGS.first_classifier_hdf5, 'r')
	# else:
	print '!!! open hdf5 file: %s' % (FLAGS.hdf5_pretrain_model)
	hdf5_data = h5py.File(FLAGS.hdf5_pretrain_model, 'r')


def close_hdf5_file():
	global hdf5_data

	if hdf5_data is not None:
		hdf5_data.close()


hdf5_data = None

def open_sports_3d_hdf5_file():
	global sports_3d_hdf5_data
	if sports_3d_hdf5_data is None:
		sports_3d_hdf5_data = h5py.File(FLAGS.hdf5_sports_3d_model, 'r')

def open_hip_hdf5_file():
	global hip_3d_hdf5_data
	if hip_3d_hdf5_data is None:
		hip_3d_hdf5_data = h5py.File(FLAGS.hdf5_hip_transfer_model, 'r')


def close_hdf5_file():
	global hdf5_data

	if hdf5_data is not None:
		hdf5_data.close()


def get_sports_3d_model_weights(layer_name, para_name):

	'''
	transfer weights from c3d model:  Learning spatiotemporal features with 3d convolutional networks in Proceedings of CVPR 2015
	'''

	global sports_3d_hdf5_data
	if sports_3d_hdf5_data is None:
		open_sports_3d_hdf5_file()

	# layer_names = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b']
	layer_index = {'conv1a':0, 'conv2a':2, 'conv3a':4, 'conv3b':5, 'conv4a':7, 'conv4b':8}
	layer_name = layer_name.replace('t2_', '')
	print '>> get_sports_3d_model_weights, layer_name=', layer_name
	ret_data = None
	if layer_name in layer_index:
		_index = layer_index[layer_name]
		index_name = 'layer_'+str(_index)
		print '>> ', type(sports_3d_hdf5_data)
		parameters = sports_3d_hdf5_data[index_name]

		if para_name == 'weights':
			ret_data = parameters['param_0']
			if layer_name == 'conv1a':
				ret_data = ret_data[:,0:1,:,:,:]
		elif para_name == 'biases':
			ret_data = parameters['param_1']
		else:
			# hdf5_data.close()
			raise Exception("para_name must be in ['weights', 'biases'] ...") 
	else:
		# hdf5_data.close()
		raise Exception("layer_name must be in %s ..." % (layer_index.keys(), )) 

	# hdf5_data.close()
	# print '>>> ret_data=',ret_data.shape
	if len(ret_data.shape)==5:
		ret_data = np.transpose(ret_data, (2,3,4,1,0))

	return np.asarray(ret_data)


def get_hip_pretrain_weights(layer_name, para_name):

	layer_index = { 't1_conv1a', 't1_conv2a', 't1_conv3a', 't1_conv3b', 't1_conv4a', 't1_conv4b', \
				 'deconv1b', 'deconv1c', \
				'deconv2a', 'deconv2b', 'deconv2c', 'deconv3a', 'deconv3b', 'deconv3c', \
				}

	global hip_3d_hdf5_data
	if hip_3d_hdf5_data is None:
		open_hip_hdf5_file()
	print '>> borrow weights: %s/%s' % (layer_name, para_name)
	ret_data = None

	if layer_name in layer_index:

		index_name = layer_name.replace('t1_', '')
		print '>> get_sports_3d_model_weights, layer_name=', index_name

		parameters = hip_3d_hdf5_data[index_name]
		
		if para_name == 'weights':
			ret_data = parameters['weights']
			
		elif para_name == 'biases':
			ret_data = parameters['biases']
		else:
			# hdf5_data.close()
			raise Exception("para_name must be in ['weights', 'biases'] ...") 

	else:
		print '>> borrow fail.. will init as randomly..'
		return None

	# hdf5_data.close()
	return np.asarray(ret_data)


def get_pretrained_weights(layer_name, para_name,shape):
	layer_index = { 't1_conv1a', 't1_conv2a', 't1_conv3a', 't1_conv3b', 't1_conv4a', 't1_conv4b', \
				't2_conv1a', 't2_conv2a', 't2_conv3a', 't2_conv3b', 't2_conv4a', 't2_conv4b', \
				 'deconv1b', 'deconv1c', \
				'deconv2a', 'deconv2b', 'deconv2c', 'deconv3a', 'deconv3b', 'deconv3c', }

	if layer_name not in layer_index:
		return None

	print '>> layer name = ', layer_name
	if 't2' in layer_name:
		# print '>>> in t2??!!'
		return get_sports_3d_model_weights(layer_name, para_name)
	else:
		# print '>>> not in t2??!!'
		return get_hip_pretrain_weights(layer_name, para_name)
	

def extract_weigths():
	from model import  create_model_infant_seg
	from main import prepare_dirs, setup_tensorflow

	prepare_dirs(delete_train_dir=False)
	sess, summary_writer = setup_tensorflow()

	(tf_t1_input, tf_t2_input, tf_label, 
			aux1_pred, aux2_pred, main_pred,
			aux1_loss, aux2_loss, main_loss, 
			final_loss, gene_vars, main_possibility) = create_model_infant_seg(train_phase=False)
	

	saver = tf.train.Saver()
	# model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	model_path = tf.train.latest_checkpoint(FLAGS.last_trained_checkpoint)
	# hdf5_data = h5py.File(FLAGS.first_classifier_hdf5, 'w')

	print('saver restore from:%s' % model_path)
	saver.restore(sess, model_path)

	print '** after resotre..'
	hdf5_data = h5py.File(FLAGS.model_saved_hdf5, 'w')
	for op in tf.trainable_variables():
		# print str(op.name), sess.run(op).shape
		# if op.name == 'main_pred/biases:0':
		# 	print sess.run(op)

		para_name = op.name.split('/')[-1]
		if para_name.startswith('weights') or para_name.startswith('biases'):
			# layer_name = op.name.split('/')[0]
			# para_name = para_name[:-2]
			print '>> create data value:',op.name[:-2]
			value = sess.run(op)
			dset = hdf5_data.create_dataset(op.name[:-2], value.shape, dtype=np.float32)
			dset[...] = value

	hdf5_data.close()



if __name__ == '__main__':

	extract_weigths()
	
	