# import tensorflow as tf
from config import FLAGS
import os.path
import numpy as np
import math
import h5py
import time
from patch_extraction import extract_test_patches
from util.utils import load_nifti, save_nifti
# from util.utils import pickle_dump, pickle_load
import nibabel as nib
import tensorflow as tf
import glob
from util.utils import parse_patch_size
from util.utils import save_hdr_img



def vote_overlapped_patch(predictions, patch_indx, d,h,w):
	'''
	Compute probability of overlapped patches
	'''

	assert len(predictions.shape)==6, 'vote_overlapped_patch, shape of predictions must be 6 '
	num_patch = predictions.shape[0]
	assert num_patch==patch_indx.shape[0], 'the first dimention of predictions and pactch_indxs must be the same..'
	cls_num = predictions.shape[-1]
	assert cls_num == FLAGS.cls_out, 'currently, only 3 class-num can be done..'
	# patch_size = FLAGS.patch_size
	patch_size = parse_patch_size(FLAGS.patch_size_str)

	print ('in vote: predictions.shape=%s' %(predictions.shape, ))

	sum_cls_all = np.zeros((d,h,w, FLAGS.cls_out))
	sum_cls_count = np.zeros((d,h,w, FLAGS.cls_out))

	for _i in xrange(num_patch):
		_pred = predictions[_i]
		_pos = patch_indx[_i]
		
		d_s, d_e = _pos[0], _pos[0]+patch_size[0]
		h_s, h_e = _pos[1], _pos[1]+patch_size[1]
		w_s, w_e = _pos[2], _pos[2]+patch_size[2]
		
		assert d_e <= d, 'd_e:%d must be less equal than d:%d' % (d_e, d)
		assert h_e <= h, 'h_e:%d must be less equal than h:%d' % (h_e, h)
		assert w_e <= w, 'w_e:%d must be less equal than w:%d' % (w_e, w)

		sum_cls_all[d_s:d_e,h_s:h_e, w_s:w_e, :] += _pred[0]
		sum_cls_count[d_s:d_e,h_s:h_e, w_s:w_e, :] += 1

	possibilty_map = sum_cls_all / sum_cls_count


	final_segmentation = np.argmax(possibilty_map, axis=-1)



	return final_segmentation, possibilty_map

def predict_multi_modality_one_img_without_label(td, t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches, index, d,h,w):
	preds_aux1 = []
	preds_aux2 = []
	preds_main = []

	start_time  = time.time()
	patch_num = t1_patches.shape[0]
	print '>> begin predict likelihood of each patch ..'
	for _i in  xrange(patch_num):
		_t1_patch = t1_patches[_i]
		_t2_patch = t2_patches[_i]
		_dm1_patch = dm1_patches[_i]
		_dm2_patch = dm2_patches[_i]
		_dm3_patch = dm3_patches[_i]
		_index = index[_i]

		feed_dict = { td.tf_t1_input : _t1_patch, 
						td.tf_t2_input : _t2_patch,
						td.tf_dm_input1 : _dm1_patch, 
						td.tf_dm_input2 : _dm2_patch, 
						td.tf_dm_input3 : _dm3_patch }
		ops = [td.aux1_pred, td.aux2_pred, td.main_possibility]
		[aux1_pred, aux2_pred, main_pred] = td.sess.run(ops, feed_dict=feed_dict)
		preds_aux1.append(aux1_pred)
		preds_aux2.append(aux2_pred)
		preds_main.append(main_pred)

	patches_pred = np.asarray(preds_main)

	print '>> begin vote in overlapped patch..'
	seg_res, possibilty_map = vote_overlapped_patch(patches_pred, index, d,h,w)
	# seconds
	elapsed = int(time.time() - start_time)

	print('predit patches of 1 iamge, cost [%3d] seconds' % (elapsed))

	return seg_res, possibilty_map


def remove_test_backgrounds(img_data, t2_data, img_data_dm1, img_data_dm2, img_data_dm3):
	if len(img_data.shape) ==4:
		img_data = img_data[:,:,:,0]
		t2_data = t2_data[:,:,:,0]
	assert len(img_data.shape)==3, 'must be 3...'
	assert len(t2_data.shape)==3, 'must be 3...'
	nonzero_label = t2_data != 0
	nonzero_label = np.asarray(nonzero_label)
	

	nonzero_index = np.nonzero(nonzero_label)
	nonzero_index = np.asarray(nonzero_index)

	x_min, x_max = nonzero_index[0,:].min(), nonzero_index[0,:].max()
	y_min, y_max = nonzero_index[1,:].min(), nonzero_index[1,:].max()
	z_min, z_max = nonzero_index[2,:].min(), nonzero_index[2,:].max()

	print '>>> remove_test_backgrounds crop: (min, max): '
	print 'img_data.shape=', img_data.shape
	print x_min, x_max
	print y_min, y_max
	print z_min, z_max

	x_min = x_min - FLAGS.prepost_pad if x_min-FLAGS.prepost_pad>=0 else 0
	y_min = y_min - FLAGS.prepost_pad if y_min-FLAGS.prepost_pad>=0 else 0
	z_min = z_min - FLAGS.prepost_pad if z_min-FLAGS.prepost_pad>=0 else 0

	x_max = x_max + FLAGS.prepost_pad if x_max+FLAGS.prepost_pad<=img_data.shape[0] else img_data.shape[0]
	y_max = y_max + FLAGS.prepost_pad if y_max+FLAGS.prepost_pad<=img_data.shape[1] else img_data.shape[1]
	z_max = z_max + FLAGS.prepost_pad if z_max+FLAGS.prepost_pad<=img_data.shape[2] else img_data.shape[2]

	print 'After patch padding crop: (min, max): '
	print x_min, x_max
	print y_min, y_max
	print z_min, z_max

	crop_index = (x_min,x_max, y_min, y_max, z_min, z_max)
	return (img_data[x_min:x_max, y_min:y_max, z_min:z_max], t2_data[x_min:x_max, y_min:y_max, z_min:z_max],\
			img_data_dm1[x_min:x_max, y_min:y_max, z_min:z_max],\
			img_data_dm2[x_min:x_max, y_min:y_max, z_min:z_max],\
			img_data_dm3[x_min:x_max, y_min:y_max, z_min:z_max],\
			crop_index)




def predict_multi_modality_img_in_nifti_path(td, t1_nifti_path, t2_nifti_path, \
									dm1_file_path, dm2_file_path, dm3_file_path, \
									save_pred_path, save_post_path, ground_truth_path=None):
	start_time  = time.time()
	print '>> begin predict nifit image: %s' % (t1_nifti_path)
	img_data_t1, nifti_img = load_nifti(t1_nifti_path)
	img_data_t2, _ = load_nifti(t2_nifti_path)
	img_data_dm1, _ = load_nifti(dm1_file_path)
	img_data_dm2, _ = load_nifti(dm2_file_path)
	img_data_dm3, _ = load_nifti(dm3_file_path)

	
	d_ori = img_data_t1.shape[0]
	h_ori = img_data_t1.shape[1]
	w_ori = img_data_t1.shape[2]
	# from preprocess import remove_test_backgrounds
	t1_data_rmbg, t2_data_rmbg, \
		dm1_data_rmbg,  dm2_data_rmbg, dm3_data_rmbg, crop_index = remove_test_backgrounds(img_data_t1, img_data_t2, \
																					img_data_dm1, img_data_dm2, img_data_dm3)
	print 'crop_index', crop_index


	t1_data_rmbg = t1_data_rmbg[np.newaxis,np.newaxis,...]
	t1_data_rmbg = np.asarray(t1_data_rmbg, dtype=np.float32)

	t2_data_rmbg = t2_data_rmbg[np.newaxis,np.newaxis,...]
	t2_data_rmbg = np.asarray(t2_data_rmbg, dtype=np.float32)

	dm1_data_rmbg = dm1_data_rmbg[np.newaxis, np.newaxis, ...]
	dm1_data_rmbg = np.asarray(dm1_data_rmbg, dtype=np.float32)

	dm2_data_rmbg = dm2_data_rmbg[np.newaxis, np.newaxis, ...]
	dm2_data_rmbg = np.asarray(dm2_data_rmbg, dtype=np.float32)

	dm3_data_rmbg = dm3_data_rmbg[np.newaxis, np.newaxis, ...]
	dm3_data_rmbg = np.asarray(dm3_data_rmbg, dtype=np.float32)

	
	t2_patches, index, d,h,w = extract_test_patches(t2_data_rmbg)
	dm1_patches, index, d,h,w = extract_test_patches(dm1_data_rmbg,normalise=False)
	dm2_patches, index, d,h,w = extract_test_patches(dm2_data_rmbg,normalise=False)
	dm3_patches, index, d,h,w = extract_test_patches(dm3_data_rmbg,normalise=False)
	t1_patches, index, d,h,w = extract_test_patches(t1_data_rmbg)
	segmentations, possibilty_map = predict_multi_modality_one_img_without_label (td, t1_patches,t2_patches, \
																				dm1_patches, dm2_patches,dm3_patches,\
																				index, d,h,w)

	segmentations = np.asarray(segmentations,  'uint8')
	assert len(segmentations.shape) == 3, '** segmentation result must be in 3-dimension'
	final_segmentation = np.zeros((d_ori,h_ori,w_ori))
	final_segmentation[crop_index[0]:crop_index[1], crop_index[2]:crop_index[3], crop_index[4]:crop_index[5]] = segmentations 
	print '**>> img_data_t2 ',img_data_t2.shape
	print 'final_segmentatio ', final_segmentation.shape
	if len(img_data_t2.shape) == 4:
		original_data = img_data_t2[:,:,:,0]
	else:
		original_data = img_data_t2

	final_segmentation[original_data==0] = 0
	
	final_segmentation = final_segmentation.astype(np.uint8)

	
	save_hdr_img(final_segmentation, nifti_img.affine, nifti_img.header, save_pred_path)
	

	elapsed = int(time.time() - start_time)

	print('!!! predit patches of 1 iamge, cost [%3d] seconds ' % (elapsed, ))



def predict_multi_modality_test_images_in_nifti(td):

	# for test_path in ['./data_Miccai2017_ISeg/iSeg-2017-Training', './data_Miccai2017_ISeg/iSeg-2017-Testing']:
	for test_path in [FLAGS.test_dir,  ]:

		
		dir_list = glob.glob('%s/*/' % (test_path,))

		for _dir in dir_list:
			file_name = _dir.split('/')[-2]
			t1_file_path = '%s/%s-T1.nii.gz' %(_dir, file_name, )
			t2_file_path = '%s/%s-T2.nii.gz' %(_dir, file_name, )
			dm1_file_path = '%s/%s_cls1_distancemap.nii.gz' %(_dir, file_name, )
			dm2_file_path = '%s/%s_cls2_distancemap.nii.gz' %(_dir, file_name, )
			dm3_file_path = '%s/%s_cls3_distancemap.nii.gz' %(_dir, file_name, )

			save_pred_path = '%s/%s_prediction_2stage.nii.gz' %(_dir, file_name, )
			save_post_path = '%s/%s_post_2stage.nii.gz' %(_dir, file_name, )

			predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, dm1_file_path, dm2_file_path, dm3_file_path, 
														save_pred_path, save_post_path)


			# break


def main():

	print '>>>> start predict...'
	predict_multi_modality_test_images_in_nifti(None)
	

if __name__ == '__main__':
	main()
	# test()