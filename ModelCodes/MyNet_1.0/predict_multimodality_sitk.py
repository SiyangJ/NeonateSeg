from config import FLAGS
import os
import numpy as np
import math
import h5py
import time
from patch_extraction import extract_test_patches
# from postprocess import post_predict
from util.utils import Dice
from util.utils import load_nifti, save_nifti
from util.utils import load_sitk, save_sitk
from util.utils import pickle_dump, pickle_load
import nibabel as nib
import tensorflow as tf
import glob
from util.utils import parse_patch_size,save_hdr_img

from generator import get_data_list
import SimpleITK as sitk


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

def predict_multi_modality_dm_one_img_without_label(td, t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches, index, d,h,w):
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

def remove_test_backgrounds(img_data, t2_data, img_data_dm1=None, img_data_dm2=None, img_data_dm3=None):
    if len(img_data.shape) ==4:
        img_data = img_data[:,:,:,0]
        t2_data = t2_data[:,:,:,0]
    assert len(img_data.shape)==3, 'must be 3...'
    assert len(t2_data.shape)==3, 'must be 3...'
    '''
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
    '''
    img_shape = img_data.shape
    x_min = y_min = z_min = 0
    x_max = img_shape[0]
    y_max = img_shape[1]
    z_max = img_shape[2]
    crop_index = (x_min,x_max, y_min, y_max, z_min, z_max)
    if FLAGS.stage_1:
        return (img_data[x_min:x_max, y_min:y_max, z_min:z_max], t2_data[x_min:x_max, y_min:y_max, z_min:z_max], crop_index)
    else:
        return (img_data[x_min:x_max, y_min:y_max, z_min:z_max], t2_data[x_min:x_max, y_min:y_max, z_min:z_max], img_data_dm1[x_min:x_max, y_min:y_max, z_min:z_max], img_data_dm2[x_min:x_max, y_min:y_max, z_min:z_max], img_data_dm3[x_min:x_max, y_min:y_max, z_min:z_max],crop_index)

def predict_multi_modality_img_in_nifti_path(td, t1_nifti_path, t2_nifti_path, save_pred_path, dm1_file_path=None, dm2_file_path=None, dm3_file_path=None):
    start_time  = time.time()
    print '>> begin predict nifit image: %s' % (t1_nifti_path)
    img_data_t1 = load_sitk(t1_nifti_path)
    img_data_t2 = load_sitk(t2_nifti_path)
    if not FLAGS.stage_1:
        img_data_dm1 = load_sitk(dm1_file_path)
        img_data_dm2 = load_sitk(dm2_file_path)
        img_data_dm3 = load_sitk(dm3_file_path)
    print '>> load nifti image finish..shape=%s' % (img_data_t1.shape, )
    
    d_ori = img_data_t1.shape[0]
    h_ori = img_data_t1.shape[1]
    w_ori = img_data_t1.shape[2]
    # from preprocess import remove_test_backgrounds
    if FLAGS.stage_1:
        t1_data_rmbg, t2_data_rmbg, crop_index = remove_test_backgrounds(img_data_t1, img_data_t2)
    else:
        t1_data_rmbg, t2_data_rmbg, dm1_data_rmbg,  dm2_data_rmbg, dm3_data_rmbg, crop_index = remove_test_backgrounds(img_data_t1, img_data_t2, img_data_dm1, img_data_dm2, img_data_dm3)
    print 'crop_index', crop_index

    t1_data_rmbg = t1_data_rmbg[np.newaxis,np.newaxis,...]
    t1_data_rmbg = np.asarray(t1_data_rmbg, dtype=np.float32)

    t2_data_rmbg = t2_data_rmbg[np.newaxis,np.newaxis,...]
    t2_data_rmbg = np.asarray(t2_data_rmbg, dtype=np.float32)
    
    if not FLAGS.stage_1:
        dm1_data_rmbg = dm1_data_rmbg[np.newaxis, np.newaxis, ...]
        dm1_data_rmbg = np.asarray(dm1_data_rmbg, dtype=np.float32)

        dm2_data_rmbg = dm2_data_rmbg[np.newaxis, np.newaxis, ...]
        dm2_data_rmbg = np.asarray(dm2_data_rmbg, dtype=np.float32)

        dm3_data_rmbg = dm3_data_rmbg[np.newaxis, np.newaxis, ...]
        dm3_data_rmbg = np.asarray(dm3_data_rmbg, dtype=np.float32)

    t1_patches, index, d,h,w = extract_test_patches(t1_data_rmbg)
    t2_patches, index, d,h,w = extract_test_patches(t2_data_rmbg)
    
    if not FLAGS.stage_1:
        dm1_patches, index, d,h,w = extract_test_patches(dm1_data_rmbg,normalise=False)
        dm2_patches, index, d,h,w = extract_test_patches(dm2_data_rmbg,normalise=False)
        dm3_patches, index, d,h,w = extract_test_patches(dm3_data_rmbg,normalise=False)
    if FLAGS.stage_1:
        segmentations, possibilty_map =  predict_multi_modality_one_img_without_label(td, t1_patches,t2_patches, index, d,h,w)
    else:
        segmentations, possibilty_map = predict_multi_modality_dm_one_img_without_label(td, t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches, index, d,h,w)

    segmentations = np.asarray(segmentations,  'uint8')
    assert len(segmentations.shape) == 3, '** segmentation result must be in 3-dimension'
    final_segmentation = np.zeros((d_ori,h_ori,w_ori))
    final_segmentation[crop_index[0]:crop_index[1], crop_index[2]:crop_index[3], crop_index[4]:crop_index[5]] = segmentations 
    print '**>> img_data_t2 ',img_data_t2.shape
    print 'final_segmentation ', final_segmentation.shape
    if len(img_data_t2.shape) == 4:
        original_data = img_data_t2[:,:,:,0]
    else:
        original_data = img_data_t2

    final_segmentation[original_data==0] = 0
    
    final_segmentation = final_segmentation.astype(np.uint8)
    
    if save_pred_path is None:
        return final_segmentation

    save_sitk(final_segmentation, save_pred_path)
    elapsed = int(time.time() - start_time)
    print('!!! predict patches of 1 image, cost [%3d] seconds ' % (elapsed, ))

## DONE editing
def extract_distance_map(input_file, bg_mask_file, ouput_file):
    print '>>> input file , %s' % (input_file)
    print '  > bg_mask_file file , %s' % (bg_mask_file)
    print '  > ouput_file file , %s' % (ouput_file)

    sitk_data = load_sitk(input_file)
    sitk_data = np.asarray(sitk_data, np.int16)
    bg_mask_data = load_sitk(bg_mask_file)
    bg_mask_data = np.asarray(bg_mask_data, np.int16)
    import scipy.ndimage as ndimage
    dis_map = ndimage.distance_transform_edt(np.logical_not(sitk_data))
    dis_map = np.asarray(dis_map, np.float32)
    # set the background = 0
    dis_map[bg_mask_data==1] = 0
    # normalise
    dis_map /= np.max(dis_map)
    save_sitk(dis_map, ouput_file)

## DONE editing
def generate_distance_map(file_name):

    pred_data = load_sitk(os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name))
    for _i in [0,1,2,3]:
        bg_mask_file = os.path.join(FLAGS.prediction_save_dir,'cls%d-%s'%(_i,file_name))
        cls_data = np.asarray(pred_data==_i, np.uint8)
        save_sitk(cls_data, bg_mask_file)

    for _i in [1,2,3]:
        input_file = os.path.join(FLAGS.prediction_save_dir,'cls%d-%s'%(_i,file_name))
        bg_mask_file = os.path.join(FLAGS.prediction_save_dir,'cls0-%s'%(file_name))
        ouput_file = os.path.join(FLAGS.prediction_save_dir,'distance_map_cls%d-%s'%(_i,file_name))
        extract_distance_map(input_file, bg_mask_file, ouput_file)

## DONE editing
def predict_multi_modality_test_images_in_sitk(td):

    pred_list = os.path.join(FLAGS.prediction_save_dir,'prediction_stage_1.list')
    if os.path.exists(pred_list):
        print('The list file to store prediction already exists: %s' % pred_list)
        os.remove(pred_list)
    
    ## Potentially also predict train and validation paths
    for list_path in [FLAGS.hdf5_test_list_path, ]:
        
        dir_list = get_data_list(list_path)

        for _dir in dir_list:
            t1_file_path = _dir[0]
            file_name = t1_file_path.split('/')[-1]
            t2_file_path = _dir[1]
            save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name)
            predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path)
            
            generate_distance_map(file_name)
            with open(pred_list,'a') as f:
                f.write(t1_file_path)
                f.write(',')
                f.write(t2_file_path)
                f.write(',')
                f.write(_dir[2])
                f.write(',')
                f.write(os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name))
                f.write(',')
                for _i,_c in zip([1,2,3],[',',',','\n']):
                    f.write(os.path.join(FLAGS.prediction_save_dir,'distance_map_cls%d-%s'%(_i,file_name)))
                    f.write(_c)
                    
def predict_multi_modality_dm_test_images_in_sitk(td):

    assert not FLAGS.stage_1, "Can only be used in Stage 2"

    pred_list = os.path.join(FLAGS.prediction_save_dir,'prediction_stage_1.list')
    if os.path.exists(pred_list):
        print('The list file to store prediction already exists: %s' % pred_list)
        os.remove(pred_list)
    for list_path in [FLAGS.hdf5_test_list_path, ]:
        
        dir_list = get_data_list(list_path)

        for _dir in dir_list:
            t1_file_path = _dir[0]
            file_name = t1_file_path.split('/')[-1]
            t2_file_path = _dir[1]
            dm1_file_path = _dir[4]
            dm2_file_path = _dir[5]
            dm3_file_path = _dir[6]
            
            save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name)
            predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path, dm1_file_path, dm2_file_path, dm3_file_path)
            with open(pred_list,'a') as f:
                for _path in _dir:
                    f.write(_path)
                    f.write(',')
                f.write(os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name))
                f.write('\n')
        print '>>> Finish predicting list %s' % list_path
    print '>>> Prediction finished!!!'

def eval_test_images_in_sitk(td):
    
    stats_list = []
    list_path = FLAGS.hdf5_test_list_path
    dir_list = get_data_list(list_path)
    for _dir in dir_list:
        t1_file_path = _dir[0]
        file_name = t1_file_path.split('/')[-1]
        t2_file_path = _dir[1]
        label_file_path = _dir[2]
        if not FLAGS.stage_1:
            dm1_file_path = _dir[4]
            dm2_file_path = _dir[5]
            dm3_file_path = _dir[6]

        if FLAGS.stage_1:
            final_segmentation = predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, None)
        else:
            final_segmentation = predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, None, dm1_file_path, dm2_file_path, dm3_file_path)

        true_label = load_sitk(label_file_path)
        stats_list += [Dice(final_segmentation,true_label),]
    
    stats_list = np.asarray(stats_list)
    stats_mean = stats_list.mean(axis=0)
    
    return stats_mean
                    
def main():
    predict_multi_modality_test_images_in_sitk(None)
    

if __name__ == '__main__':
    main()
    # test()