from config import FLAGS
import os.path
import numpy as np
import math
import h5py
import time
from patch_extraction import extract_test_patches
# from postprocess import post_predict
from util.utils import load_nifti, save_nifti
from util.utils import pickle_dump, pickle_load
import nibabel as nib
import tensorflow as tf
import glob
from util.utils import parse_patch_size,save_hdr_img


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



def predict_multi_modality_one_img_without_label(td, t1_patches, t2_patches, index, d,h,w):
	preds_aux1 = []
	preds_aux2 = []
	preds_main = []

	start_time  = time.time()
	patch_num = t1_patches.shape[0]
	print '>> begin predict likelihood of each patch ..'
	for _i in  xrange(patch_num):
		_t1_patch = t1_patches[_i]
		_t2_patch = t2_patches[_i]
		_index = index[_i]

		feed_dict = { td.tf_t1_input : _t1_patch, 
						td.tf_t2_input : _t2_patch,
					}
		ops = [td.aux1_pred, td.aux2_pred, td.main_pred]
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



def remove_test_backgrounds(img_data, t2_data):
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

	print 'Before patch padding crop: (min, max): '
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
	return (img_data[x_min:x_max, y_min:y_max, z_min:z_max], t2_data[x_min:x_max, y_min:y_max, z_min:z_max], crop_index)


def predict_multi_modality_img_in_nifti_path(td, t1_nifti_path, t2_nifti_path, save_pred_path):
	start_time  = time.time()
	print '>> begin predict nifit image: %s' % (t1_nifti_path)
	img_data_t1, nifti_img = load_nifti(t1_nifti_path)
	img_data_t2, _ = load_nifti(t2_nifti_path)
	print '>>!!! at start: img_data_t2 ', t2_nifti_path

	print '>> load nifti image finish..shape=%s' % (img_data_t1.shape, )
	
	d_ori = img_data_t1.shape[0]
	h_ori = img_data_t1.shape[1]
	w_ori = img_data_t1.shape[2]
	# from preprocess import remove_test_backgrounds
	t1_data_rmbg, t2_data_rmbg, crop_index = remove_test_backgrounds(img_data_t1, img_data_t2)
	print 'crop_index', crop_index

	t1_data_rmbg = t1_data_rmbg[np.newaxis,np.newaxis,...]
	t1_data_rmbg = np.asarray(t1_data_rmbg, dtype=np.float32)

	t2_data_rmbg = t2_data_rmbg[np.newaxis,np.newaxis,...]
	t2_data_rmbg = np.asarray(t2_data_rmbg, dtype=np.float32)

	t1_patches, index, d,h,w = extract_test_patches(t1_data_rmbg)
	t2_patches, index, d,h,w = extract_test_patches(t2_data_rmbg)
	segmentations, possibilty_map =  predict_multi_modality_one_img_without_label(td, t1_patches,t2_patches, index, d,h,w)

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



def extract_distance_map(input_file, bg_mask_file, ouput_file):
	print '>>> input file , %s' % (input_file)
	print '  > bg_mask_file file , %s' % (bg_mask_file)
	print '  > ouput_file file , %s' % (ouput_file)

	nifti_data, nifti_img = load_nifti(input_file)
	nifti_data = np.asarray(nifti_data, np.int16)
	bg_mask_data, _ = load_nifti(bg_mask_file)
	bg_mask_data = np.asarray(bg_mask_data, np.int16)
	import scipy.ndimage as ndimage
	dis_map = ndimage.distance_transform_edt(np.logical_not(nifti_data))
	dis_map = np.asarray(dis_map, np.float32)
	# set the background = 0
	dis_map[bg_mask_data==1] = 0
	# normalise
	dis_map /= np.max(dis_map)
	save_nifti(dis_map, nifti_img.affine, ouput_file)

def generate_distance_map(test_path):
	sub_dirs = glob.glob("%s/*/" %(test_path, ))

	for _dir in sub_dirs:
		# print _dir
		file_name = _dir.split('/')[-2]
		save_pred_path = '%s/%s_prediction.nii.gz' %(_dir, file_name, )
		pred_data, _img = load_nifti(save_pred_path)
		for _i in [0,1,2,3]:
			bg_mask_file = '%s/%s_cls%d.nii.gz' %(_dir, file_name, _i,)
			cls_data = np.asarray(pred_data==_i, np.uint8)
			save_nifti(cls_data, _img.affine, bg_mask_file)


		for _i in [1,2,3]:
			input_file = '%s/%s_cls%d.nii.gz' %(_dir, file_name, _i)
			bg_mask_file = '%s/%s_cls0.nii.gz' %(_dir, file_name)
			ouput_file = '%s/%s_cls%d_distancemap.nii.gz' %(_dir, file_name, _i)
			
			extract_distance_map(input_file, bg_mask_file, ouput_file)

def predict_multi_modality_test_images_in_nifti(td):

	# test_path = FLAGS.test_dir
	# for test_path in [FLAGS.train_data_dir, FLAGS.test_dir]:
	for test_path in [FLAGS.test_dir, ]:
		# save_path = FLAGS.test_save_dir

		
		dir_list = glob.glob('%s/*/' % (test_path,))

		for _dir in dir_list:
			file_name = _dir.split('/')[-2]
			t1_file_path = '%s/%s-T1.nii.gz' %(_dir, file_name, )
			t2_file_path = '%s/%s-T2.nii.gz' %(_dir, file_name, )

			save_pred_path = '%s/%s_prediction.nii.gz' %(_dir, file_name, )
			# save_post_path = '%s/%s_post.nii.gz' %(_dir, file_name, )


			predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path,  save_pred_path)
			# break
			
		generate_distance_map(test_path)



def main():
	predict_multi_modality_test_images_in_nifti(None)
	

if __name__ == '__main__':
	main()
	# test()