from config import FLAGS
import os
import numpy as np
import math
import h5py
import time
from patch_extraction import extract_test_patches,extract_overlapped_patches_index
# from postprocess import post_predict
from util.utils import Dice
from util.utils import load_nifti, save_nifti
from util.utils import load_sitk, save_sitk
from util.utils import pickle_dump, pickle_load
import nibabel as nib
import tensorflow as tf
import glob
from util.utils import parse_patch_size,save_hdr_img,parse_string_to_numbers

from generator import get_data_list
import SimpleITK as sitk

def get_best_batch():
    best_batch_file = os.path.join(FLAGS.checkpoint_dir, 'best_batch')
    with open(best_batch_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return int(content[1])

def vote_overlapped_patch(predictions, patch_index, d,h,w,return_intermediate=False):
    '''
    Compute probability of overlapped patches
    '''
    assert len(predictions.shape)==6, 'vote_overlapped_patch, shape of predictions must be 6 '
    num_patch = predictions.shape[0]
    print 'num_patch=%d\npatch_index.shape[0]=%d' % (num_patch,patch_index.shape[0])
    assert num_patch==patch_index.shape[0], 'the first dimention of predictions and patch_index must be the same..'
    cls_num = predictions.shape[-1]
    assert cls_num == FLAGS.cls_out, 'currently, only 3 class-num can be done..'
    # patch_size = FLAGS.patch_size
    patch_size = parse_patch_size(FLAGS.patch_size_str)

    print ('in vote: predictions.shape=%s' %(predictions.shape, ))

    sum_cls_all = np.zeros((d,h,w, FLAGS.cls_out))
    sum_cls_count = np.zeros((d,h,w, FLAGS.cls_out))

    for _i in xrange(num_patch):
        _pred = predictions[_i]
        _pos = patch_index[_i]

        d_s, d_e = _pos[0], _pos[0]+patch_size[0]
        h_s, h_e = _pos[1], _pos[1]+patch_size[1]
        w_s, w_e = _pos[2], _pos[2]+patch_size[2]
        
        assert d_e <= d, 'd_e:%d must be less equal than d:%d' % (d_e, d)
        assert h_e <= h, 'h_e:%d must be less equal than h:%d' % (h_e, h)
        assert w_e <= w, 'w_e:%d must be less equal than w:%d' % (w_e, w)

        sum_cls_all[d_s:d_e,h_s:h_e, w_s:w_e, :] += _pred[0]
        sum_cls_count[d_s:d_e,h_s:h_e, w_s:w_e, :] += 1

    if return_intermediate:
        return sum_cls_all,sum_cls_count
    
    possibilty_map = sum_cls_all / sum_cls_count
    final_segmentation = np.argmax(possibilty_map, axis=-1)
    return final_segmentation, possibilty_map

def _predict_input_patches_without_label(td,all_input_patch_list,index, d,h,w,
                                         stage_1,
                                         train_phase=False,
                                         return_intermediate=False):
    preds_main = []
    patch_num = all_input_patch_list[0].shape[0]
    print '>> begin predict likelihood of each patch; total number: %d' % patch_num
    batch_size = FLAGS.batch_size if train_phase else 1
    batch_num = int(np.ceil(float(patch_num) / batch_size))
        
    for _i in  xrange(batch_num):
        _to_select = np.arange(_i*batch_size,(_i+1)*batch_size)%patch_num
        _input_patch = [ np.squeeze(all_patch[_to_select],1)
                        for all_patch in all_input_patch_list]

        if stage_1:
            feed_dict = { td.tf_t1_input : _input_patch[0], 
                            td.tf_t2_input : _input_patch[1]}
        else:
            feed_dict = { td.tf_t1_input : _input_patch[0], 
                            td.tf_t2_input : _input_patch[1],
                            td.tf_dm_input1 : _input_patch[2], 
                            td.tf_dm_input2 : _input_patch[3], 
                            td.tf_dm_input3 : _input_patch[4] }
        ops = [td.aux1_pred, td.aux2_pred, td.main_possibility]
        [aux1_pred, aux2_pred, main_pred] = td.sess.run(ops, feed_dict=feed_dict)
        preds_main.append(main_pred)

    preds_main = tuple(preds_main)
    patches_pred = np.vstack(preds_main)
    patches_pred = patches_pred[:patch_num,:]
    patches_pred = np.expand_dims(patches_pred,axis=1)

    print '>> begin vote in overlapped patch..'
    seg_res, possibilty_map = vote_overlapped_patch(patches_pred, index, d,h,w,return_intermediate=return_intermediate)

    return seg_res, possibilty_map

def predict_multi_modality_one_img_without_label(td, t1_patches, t2_patches, index, d,h,w,
                                                 train_phase=False,
                                                 return_intermediate=False):
    all_input_patch_list = [t1_patches, t2_patches,]
    seg_res, possibilty_map = _predict_input_patches_without_label(td,all_input_patch_list,index, d,h,w,
                                                                   True,
                                                                   train_phase=train_phase,
                                                                   return_intermediate=return_intermediate)
    return seg_res, possibilty_map

def predict_multi_modality_dm_one_img_without_label(td, t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches, index, d,h,w,train_phase=False,return_intermediate=False):
    
    all_input_patch_list = [t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches]
    seg_res, possibilty_map = _predict_input_patches_without_label(td,all_input_patch_list,index, d,h,w,
                                                                   False,
                                                                   train_phase=train_phase,
                                                                   return_intermediate=return_intermediate)
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

def predict_multi_modality_img_in_nifti_path(td, t1_nifti_path, t2_nifti_path, save_pred_path, dm1_file_path=None, dm2_file_path=None, dm3_file_path=None,train_phase=False):
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

        
    #### Editted Mar 6
    ## Enable predicting all patches in several rounds to avoid OOM
    index, d, h, w = extract_overlapped_patches_index(t1_data_rmbg)
    num_patches = index.shape[0]
    num_iters = int(np.ceil(float(num_patches) / FLAGS.max_patch_num))
    
    sum_cls_all_arr = []
    sum_cls_count_arr = []
    
    for _i in xrange(num_iters):
        
        cur_patch_index_low = _i * FLAGS.max_patch_num
        cur_patch_index_high = min((_i+1) * FLAGS.max_patch_num, num_patches)
        
        cur_patch_index = index[cur_patch_index_low : cur_patch_index_high]
    
        t1_patches, _,_,_,_ = extract_test_patches(t1_data_rmbg,patches_index=cur_patch_index)
        t2_patches, _,_,_,_ = extract_test_patches(t2_data_rmbg,patches_index=cur_patch_index)

        if not FLAGS.stage_1:
            dm1_patches, _,_,_,_ = extract_test_patches(dm1_data_rmbg,normalise=False,patches_index=cur_patch_index)
            dm2_patches, _,_,_,_ = extract_test_patches(dm2_data_rmbg,normalise=False,patches_index=cur_patch_index)
            dm3_patches, _,_,_,_ = extract_test_patches(dm3_data_rmbg,normalise=False,patches_index=cur_patch_index)
        if FLAGS.stage_1:
            segmentations, possibilty_map =  predict_multi_modality_one_img_without_label(td, t1_patches,t2_patches, 
                                                                                          cur_patch_index, d,h,w,
                                                                                          train_phase=train_phase,
                                                                                          return_intermediate=True)
        else:
            segmentations, possibilty_map = predict_multi_modality_dm_one_img_without_label(td, t1_patches, t2_patches,dm1_patches, dm2_patches,dm3_patches, cur_patch_index, d,h,w,train_phase=train_phase,return_intermediate=True)
            
        sum_cls_all_arr   += [segmentations,]
        sum_cls_count_arr += [possibilty_map,]
    
    sum_cls_all = np.asarray(sum_cls_all_arr).sum(axis=0)
    sum_cls_count = np.asarray(sum_cls_count_arr).sum(axis=0)
    
    possibilty_map = sum_cls_all / sum_cls_count
    segmentations = np.argmax(possibilty_map, axis=-1)

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

def generate_error_map(pred_path,true_path,file_name,prediction_save_dir=FLAGS.prediction_save_dir):
    
    output_file = os.path.join(prediction_save_dir,'error_map-%s'%(file_name))
    print '>>> Generating error map: %s' % (file_name)
    print '  > prediction: %s' % (pred_path)
    print '  > ground truth: %s' % (true_path)
    print '  > ouput file: %s' % (output_file)
    pred_data = load_sitk(pred_path)
    true_data = load_sitk(true_path)
    
    raw_error_map = (pred_data!=true_data).astype(np.float32)
    
    ## TODO Consider different filtering.
    from scipy.ndimage.filters import convolve
    kernel_size = FLAGS.error_map_kernel_size
    if FLAGS.error_map_kernel == 'ones':
        error_filter = np.full((kernel_size,kernel_size,kernel_size),1)
        error_map = convolve(raw_error_map,error_filter,mode='constant',cval=0.0)
    else:
        print 'Not implemented'
    
    error_map[error_map==0] = FLAGS.error_map_correct_weight
    
    save_sitk(error_map, output_file)

    
def regenerate_error_map(prediction_save_dir=FLAGS.prediction_save_dir,new_prediction_file=False):
    pred_list = os.path.join(prediction_save_dir,'prediction_stage_1.list')
    assert os.path.exists(pred_list),'The list file to store prediction does not exist: %s' % pred_list

    dir_list = get_data_list(pred_list)
    if new_prediction_file:
        os.remove(pred_list)
    
    for _dir in dir_list:
        t1_file_path = _dir[0]
        file_name = t1_file_path.split('/')[-1]
        
        save_pred_path = _dir[3]
        truth_path = _dir[2]
        generate_error_map(save_pred_path,truth_path,file_name,prediction_save_dir=prediction_save_dir)

        if new_prediction_file:
            with open(pred_list,'a') as f:
                for _i in xrange(7):
                    f.write(_dir[_i])
                    f.write(',')
                f.write(os.path.join(prediction_save_dir,'error_map-%s'%(file_name)))
                f.write('\n')
                
    print '>>> Finish regenerating error maps <<<'
    
def split_data(test_num=1,val_num=8,data_list=None):
    if data_list is None:
        data_list = os.path.join(FLAGS.prediction_save_dir,'prediction_stage_1.list')
    assert os.path.exists(data_list),'The list file to split does not exist: %s' % data_list

    dir_list = get_data_list(data_list)
    for _dir in dir_list:
        t1_file_path = _dir[0]
        file_name = t1_file_path.split('/')[-1]
        
        if str(test_num) in file_name:
            split_name = 'test'
        elif str(val_num) in file_name:
            split_name = 'validation'
        else:
            split_name = 'train'
            
        cur_list = data_list[:-5] + '_' + split_name + '.list'
        with open(cur_list,'a') as f:
            _dir_len = len(_dir)
            for _i in xrange(_dir_len):
                f.write(_dir[_i])
                if _i<_dir_len-1:
                    f.write(',')
                else:
                    f.write('\n')
    
    print '>>> Finish splitting data <<<'
    
## 
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
            print t1_file_path, t2_file_path
            if not FLAGS.save_around_best:
                save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name)
                predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path)
            else:
                saver = tf.train.Saver()
                best_batch = get_best_batch()
                print best_batch
                
                im_size = load_sitk(t1_file_path).shape
                output_image = np.zeros(im_size + (4,))
                
                prediction_save_dir = FLAGS.prediction_save_dir
                
                for _i in xrange(-FLAGS.save_around_num, FLAGS.save_around_num+1):
                    if _i ==0:
                        batch = 'best'
                    else:
                        batch = best_batch + _i * FLAGS.validate_every_n
                    
                    model_path = os.path.join(FLAGS.checkpoint_dir, 'snapshot_%s'%batch)
                    print('saver restore from:%s' % model_path)
                    saver.restore(td.sess, model_path)

                    FLAGS.prediction_save_dir = os.path.join(prediction_save_dir, 'batch_%s'%batch)
                    if not os.path.exists(FLAGS.prediction_save_dir):
                        os.mkdir(FLAGS.prediction_save_dir)
                        
                    save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name)
                    predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path)
                    
                    cur_image = load_sitk(save_pred_path)
                    for _l in xrange(FLAGS.cls_out):
                        output_image[:,:,:,_l] += (cur_image==_l).astype(int)
                
                output_image = np.argmax(output_image,axis=-1)
                FLAGS.prediction_save_dir = prediction_save_dir
                save_sitk(output_image,os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name))
            
            generate_distance_map(file_name)
            if FLAGS.output_error_map:
                truth_path = _dir[2]
                generate_error_map(save_pred_path,truth_path,file_name)
            
            with open(pred_list,'a') as f:
                f.write(t1_file_path)
                f.write(',')
                f.write(t2_file_path)
                f.write(',')
                f.write(_dir[2])
                f.write(',')
                f.write(os.path.join(FLAGS.prediction_save_dir,'prediction-'+file_name))
                f.write(',')
                for _i,_c in zip([1,2,3],[',',',','']):
                    f.write(os.path.join(FLAGS.prediction_save_dir,'distance_map_cls%d-%s'%(_i,file_name)))
                    f.write(_c)
                if FLAGS.output_error_map:
                    f.write(',')
                    f.write(os.path.join(FLAGS.prediction_save_dir,'error_map-%s'%(file_name)))
                f.write('\n')
        print '>>> Finish predicting list %s' % list_path
    print '>>> Prediction finished!!!'
    if FLAGS.split_data_after_test:
        split_data(data_list=pred_list)
        
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
            
            '''
            save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name)
            predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path, dm1_file_path, dm2_file_path, dm3_file_path)
            '''
            
            if not FLAGS.save_around_best:
                save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name)
                predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path, dm1_file_path, dm2_file_path, dm3_file_path)
            else:
                saver = tf.train.Saver()
                best_batch = get_best_batch()
                print 'best_batch=%s'%best_batch
                
                im_size = load_sitk(t1_file_path).shape
                output_image = np.zeros(im_size + (4,))
                
                prediction_save_dir = FLAGS.prediction_save_dir
                
                for _i in xrange(-FLAGS.save_around_num, FLAGS.save_around_num+1):
                    if _i ==0:
                        batch = 'best'
                    else:
                        batch = best_batch + _i * FLAGS.validate_every_n
                    
                    model_path = os.path.join(FLAGS.checkpoint_dir, 'snapshot_%s'%batch)
                    print('saver restore from:%s' % model_path)
                    saver.restore(td.sess, model_path)

                    FLAGS.prediction_save_dir = os.path.join(prediction_save_dir, 'batch_%s'%batch)
                    if not os.path.exists(FLAGS.prediction_save_dir):
                        os.mkdir(FLAGS.prediction_save_dir)
                        
                    save_pred_path = os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name)
                    predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, save_pred_path, dm1_file_path, dm2_file_path, dm3_file_path)
                    
                    cur_image = load_sitk(save_pred_path)
                    for _l in xrange(FLAGS.cls_out):
                        output_image[:,:,:,_l] += (cur_image==_l).astype(int)
                
                output_image = np.argmax(output_image,axis=-1)
                FLAGS.prediction_save_dir = prediction_save_dir
                save_sitk(output_image,os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name))
            
            with open(pred_list,'a') as f:
                for _path in _dir:
                    f.write(_path)
                    f.write(',')
                f.write(os.path.join(FLAGS.prediction_save_dir,'prediction-2-'+file_name))
                f.write('\n')
        print '>>> Finish predicting list %s' % list_path
    print '>>> Prediction finished!!!'
    
def eval_test_images_in_sitk(td,train_phase=True,_debug=False):
    
    cls_labels = list(parse_string_to_numbers(FLAGS.cls_labels,to_type=int))
    assert len(cls_labels)==FLAGS.cls_out, "Number of classes don't match"
    
    stats_list = []
    
    list_path = FLAGS.hdf5_test_list_path
    dir_list = get_data_list(list_path)
    
    if _debug:
        final_seg_list = []
        true_label_list = []
    
    for _dir in dir_list:
        t1_file_path = _dir[0]
        file_name = t1_file_path.split('/')[-1]
        t2_file_path = _dir[1]
        label_file_path = _dir[2]
        print '>>> Begin evaluating with ground truth: %s' % (label_file_path)
        if not FLAGS.stage_1:
            dm1_file_path = _dir[4]
            dm2_file_path = _dir[5]
            dm3_file_path = _dir[6]

        if FLAGS.stage_1:
            final_segmentation = predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, None, train_phase=train_phase)
        else:
            final_segmentation = predict_multi_modality_img_in_nifti_path(td, t1_file_path, t2_file_path, None, dm1_file_path, dm2_file_path, dm3_file_path,train_phase=train_phase)

        true_label = load_sitk(label_file_path)
        
        if _debug:
            final_seg_list += [final_segmentation,]
            true_label_list += [true_label,]
        
        
        stats_list += [Dice(final_segmentation,true_label,labels=cls_labels),]
    
    stats_list = np.asarray(stats_list)
    stats_mean = stats_list.mean(axis=0)
    if _debug:
        return stats_list, true_label_list, final_seg_list
    return stats_mean
                    
def main():
    predict_multi_modality_test_images_in_sitk(None)
    

if __name__ == '__main__':
    main()
    # test()