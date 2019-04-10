import os
from random import shuffle
import random
import sys
import numpy as np
from copy import deepcopy
from config import FLAGS

if FLAGS.load_with_sitk:
    import SimpleITK as sitk
    from util.utils import load_sitk, save_sitk

import h5py
import math
from util.utils import pickle_dump, pickle_load
from util.utils import parse_patch_size, parse_string_to_numbers
from util.utils import load_nifti, save_nifti
def get_training_and_testing_generators(hdf5_train_list_file=FLAGS.hdf5_train_list_path,
                                        hdf5_validation_list_file=FLAGS.hdf5_validation_list_path,
                                         batch_size=FLAGS.batch_size,
                                        overwrite_split=False):
    '''
    after split the training and testing , the split will be stored as pkl.
    overwrite_split: True is to overwrite the pkl file, which states what the 
    trainging and testing hdf5 file are
    '''

    training_list, validation_list = get_validation_split( hdf5_train_list_file, hdf5_validation_list_file,
                                                        overwrite_split=overwrite_split)
    
    training_generator = data_random_generator(training_list, batch_size=batch_size)
    validation_generator = data_random_generator(validation_list, batch_size=batch_size, for_training=False)
    
    print '>>> Generating training data from %s' % hdf5_train_list_file
    print '>>> Generating validation data from %s' % hdf5_validation_list_file
    
    return training_generator, validation_generator

def _get_patch_center():
    patch_size = parse_patch_size(FLAGS.patch_size_str)
    center = [int(s/2) for s in patch_size]
    return center
'''
def _get_deformation_transform(center=_get_patch_center()):

    return bsp
'''
    
def _get_scaling_transform(center=_get_patch_center()):
    if FLAGS.scaling_percentage is None:
        return None
    scaling_percentage = np.asarray(parse_string_to_numbers(FLAGS.scaling_percentage,to_type=float),dtype=float)
    scaling_percentage = np.random.uniform(-scaling_percentage,scaling_percentage) / 100
    scaling_params = 1 + scaling_percentage
    scl = sitk.ScaleTransform(3,scaling_params.tolist())
    scl.SetCenter(center)
    return scl
    
def _get_rotation_transform(center=_get_patch_center()):
    if FLAGS.rotation_degree is None:
        return None
    rotation_degree = np.asarray(parse_string_to_numbers(FLAGS.rotation_degree,to_type=float),dtype=float)
    ## TODO: How to determine angle of rotation?
    rotation_degree = np.random.uniform(-rotation_degree,rotation_degree)
    rotation_radian = rotation_degree * np.pi/180
    rot = sitk.Euler3DTransform()
    rot.SetRotation(rotation_radian[0],rotation_radian[1],rotation_radian[2])
    rot.SetCenter(center)
    return rot
    
def _get_flip_params():
    if FLAGS.flip is None:
        return None
    flp = np.asarray(parse_string_to_numbers(FLAGS.flip,to_type=int),dtype=bool)
    rnd_flp = np.random.choice([True,False],3)
    flp = np.logical_and(flp,rnd_flp)
    return flp.tolist()

def data_augment_transform(im_T1,im_T2,im_label,return_array=True,im_pred=None,im_weight=None):
    ## TODO
    im_T1.SetOrigin([0,0,0])
    im_T2.SetOrigin([0,0,0])
    im_label.SetOrigin([0,0,0])
    if im_pred is not None:
        im_pred.SetOrigin([0,0,0])
    #center = _get_patch_center()
    center = [int(_s/2) for _s in im_T1.GetSize()]
    
    flp = _get_flip_params()
    if flp is not None:
        im_T1 = sitk.Flip(im_T1,deepcopy(flp))
        im_T2 = sitk.Flip(im_T2,deepcopy(flp))
        im_label = sitk.Flip(im_label,deepcopy(flp))
        if im_pred is not None:
            im_pred = sitk.Flip(im_pred)
    
    # Composite
    cmp = sitk.Transform(3, sitk.sitkComposite)
            
    # BSpline, learned from niftynet
    if FLAGS.deformation == True:
        num_cpt = FLAGS.num_control_points
        d_sigma = FLAGS.deformation_sigma
        trans_from_domain_mesh_size = [num_cpt] * 3
        bsp = sitk.BSplineTransformInitializer(im_T1, trans_from_domain_mesh_size)
        params = bsp.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(params_numpy.shape[0]) * d_sigma
        params = tuple(params_numpy)
        bsp.SetParameters(params)
        cmp.AddTransform(bsp)
    
    # Scaling
    scl = _get_scaling_transform(center)
    if scl is not None:
        cmp.AddTransform(scl)

    # Rotation
    rot = _get_rotation_transform(center)
    if rot is not None:
        cmp.AddTransform(rot)
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetReferenceImage(im_T1)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(cmp)
    
    arr_T1 = np.swapaxes(sitk.GetArrayFromImage(resampler.Execute(im_T1)),0,2)
    arr_T2 = np.swapaxes(sitk.GetArrayFromImage(resampler.Execute(im_T2)),0,2)
    
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    arr_label = np.swapaxes(sitk.GetArrayFromImage(resampler.Execute(im_label)),0,2)
    ret_arrs = (arr_T1,arr_T2,arr_label)
    if im_pred is not None:
        arr_pred = np.swapaxes(sitk.GetArrayFromImage(resampler.Execute(im_pred)),0,2)
        ret_arrs += (arr_pred,)
    if im_weight is not None:
        arr_weight = np.swapaxes(sitk.GetArrayFromImage(resampler.Execute(im_weight)),0,2)
        ret_arrs += (arr_weight,)
    return ret_arrs

def data_augment_transform_dm(im_T1,im_T2,im_label,im_pred,return_array=True,im_weight=None):
    
    ret_arrs = data_augment_transform(im_T1,im_T2,im_label,return_array=return_array,im_pred=im_pred,im_weight=im_weight)
    if im_weight is not None:
        arr_weight = ret_arrs[-1]
        ret_arrs = ret_arrs[:-1]
    import scipy.ndimage as ndimage
    bg_mask_data = np.asarray(ret_arrs[3]==0, np.uint8)
    for _i in [1,2,3]:
        cls_data = np.asarray(ret_arrs[3]==_i, np.uint8)
        dis_map = ndimage.distance_transform_edt(np.logical_not(cls_data))
        dis_map = np.asarray(dis_map, np.float32)
        # set the background = 0
        dis_map[bg_mask_data==1] = 0
        # normalise
        dis_map /= np.max(dis_map)
        ret_arrs += (dis_map,)
    if im_weight is not None:
        ret_arrs += (arr_weight,)
    # ret_arrs = arr_T1,arr_T2,arr_label,arr_pred,dis_map1,dis_map2,dis_map3,(arr_weight)
    return ret_arrs

def data_random_generator(hdf5_list, 
                            patch_size_str=FLAGS.patch_size_str, 
                            batch_size=1,
                            extract_batches_one_image=FLAGS.batches_one_image,
                            for_training=True):
    '''
        randome crop patches from volume images. hdf5_data contains several volume datas.
        hdf5_data: num*1*Depth*H*W
        random strategy: 0. each time extract a batch_size from 
    '''
    patch_size = parse_patch_size(patch_size_str)
        
    ################ Preload Data
    if FLAGS.preload_data and FLAGS.load_with_sitk:
        if for_training and FLAGS.augmentation:
            all_data = [[sitk.ReadImage(_local_file_idx)
                         for _local_file_idx in _local_file] 
                        for _local_file in hdf5_list]
        else:
            all_data = [[load_sitk(_local_file_idx)
                         for _local_file_idx in _local_file] 
                        for _local_file in hdf5_list]            
        print '>>> Finished **Preloading Data**'
    
    _epoch = -1
    while True:
        _epoch += 1
        if for_training and FLAGS.augmentation and _epoch % FLAGS.augmentation_per_n_epoch==0:
            ## Construct an array of augmented images
            if FLAGS.stage_1:
                if not FLAGS.use_error_map:
                    augmented_data = [list(data_augment_transform(_original_image[0],
                                                                  _original_image[1],
                                                                  _original_image[2],
                                                                  return_array=True))
                                      for _original_image in all_data]
                else:
                    augmented_data = [list(data_augment_transform(_original_image[0],
                                                                  _original_image[1],
                                                                  _original_image[2],
                                                                  _original_image[-1],
                                                                  return_array=True))
                                      for _original_image in all_data]
            else:
                if not FLAGS.use_error_map:
                    augmented_data = [list(data_augment_transform_dm(_original_image[0],
                                                                  _original_image[1],
                                                                  _original_image[2],
                                                                  _original_image[3],
                                                                  return_array=True))
                                      for _original_image in all_data]
                else:
                    augmented_data = [list(data_augment_transform_dm(_original_image[0],
                                                                  _original_image[1],
                                                                  _original_image[2],
                                                                  _original_image[3],
                                                                  _original_image[-1],
                                                                  return_array=True))
                                      for _original_image in all_data]
            print '>>> Finished **Data Augmentation the %d th time**' % _epoch
        
        if for_training and FLAGS.augmentation:
            shuffle(augmented_data)
        elif FLAGS.preload_data and FLAGS.load_with_sitk:
            shuffle(all_data)
        else:
            shuffle(hdf5_list)
        
        for _local_file in (augmented_data if (for_training and FLAGS.augmentation) else \
                            (all_data if FLAGS.preload_data and FLAGS.load_with_sitk 
                             else hdf5_list)):            
            if FLAGS.load_with_sitk:
                if FLAGS.preload_data:
                    img_data_t1,img_data_t2,img_label = _local_file[:3]                    
                    if not FLAGS.stage_1:
                        img_dm1,img_dm2,img_dm3 = _local_file[4:7]
                        img_dm_list = [img_dm1,img_dm2,img_dm3]
                    if FLAGS.use_error_map:
                        img_weight = _local_file[-1]
                else:
                    img_data_t1 = load_sitk(_local_file[0])
                    img_data_t2 = load_sitk(_local_file[1])
                    img_label   = load_sitk(_local_file[2])
                    if FLAGS.use_error_map:
                        img_weight = load_sitk(_local_file[-1])
                    if not FLAGS.stage_1:
                        img_dm_list = [np.swapaxes(sitk.GetArrayFromImage(
                            sitk.ReadImage(_local_file[_i])),0,2) for _i in [4,5,6]]
                    
            else:
                assert not FLAGS.use_error_map
                #print ('generate random patch from file %s ...' % _local_file)
                file_handle   = h5py.File(_local_file, 'r')
                img_data_t1 = np.asarray(file_handle['t1data'],'float')
                img_data_t2 = np.asarray(file_handle['t2data'],'float')
                img_label   = np.asarray(file_handle['label'],'float')
                if not FLAGS.stage_1:
                    img_dm_list = [np.asarray(file_handle[_s],'float')for s in ['dm1','dm2','dm3']]
                file_handle.close()
            
            #print '>> img_data_t1.shape=',img_data_t1.shape

            img_data_t1 = img_data_t1[np.newaxis, np.newaxis, ... ]
            img_data_t2 = img_data_t2[np.newaxis, np.newaxis, ... ]
            img_label = img_label[np.newaxis, np.newaxis, ... ]
            if FLAGS.use_error_map:
                img_weight = img_weight[np.newaxis, np.newaxis, ... ]
            if not FLAGS.stage_1:
                img_dm_list = [_dm[np.newaxis, np.newaxis, ... ]for _dm in img_dm_list]
            
            assert len(img_label.shape)==5, 'label must be in 5 dimentional..'
            assert len(img_data_t1.shape)==5, ' dimentional of volume image data must be 5..'

            d = img_data_t1.shape[-3]
            h = img_data_t1.shape[-2]
            w = img_data_t1.shape[-1]
            crop_pad = FLAGS.training_crop_pad

            #print '>> crop center [%d:-%d]... d=%d,h=%d,w=%d' %(crop_pad,crop_pad,d,h,w,)
            # how many times that we extract a batch of patches in one image
            ##print '>> Starting to generate patches from a new image for %s' % ('training' if for_training else 'validation')
            for _ in xrange(extract_batches_one_image):
                x1_list = list()
                x2_list = list()
                y_list = list()
                weight_list = list() if FLAGS.use_error_map else None
                if not FLAGS.stage_1:
                    dm_lists = [list(),list(),list()]
                for _ in xrange(batch_size):
                    d_ran = random.randrange(crop_pad,d - patch_size[0]-crop_pad+1)
                    h_ran = random.randrange(crop_pad,h - patch_size[1]-crop_pad+1)
                    w_ran = random.randrange(crop_pad,w - patch_size[2]-crop_pad+1)

                    # print ('>> random crop from (%s,%s,%s) ' % (d_ran, h_ran, w_ran))
                    random_crop_data_t1 = img_data_t1[0,0,d_ran : d_ran+patch_size[0], 
                                            h_ran : h_ran+patch_size[1],
                                            w_ran: w_ran+patch_size[2]]
                    random_crop_data_t1 = np.asarray(random_crop_data_t1)

                    random_crop_data_t2 = img_data_t2[0,0,d_ran : d_ran+patch_size[0], 
                                            h_ran : h_ran+patch_size[1],
                                            w_ran: w_ran+patch_size[2]]
                    random_crop_data_t2 = np.asarray(random_crop_data_t2)

                    random_crop_truth = img_label[0,0, d_ran : d_ran+patch_size[0], 
                                            h_ran : h_ran+patch_size[1],
                                            w_ran: w_ran+patch_size[2]]
                    random_crop_truth = np.asarray(random_crop_truth)
                    if FLAGS.use_error_map:
                        random_crop_weight = img_weight[0,0,d_ran : d_ran+patch_size[0], 
                                                h_ran : h_ran+patch_size[1],
                                                w_ran: w_ran+patch_size[2]]
                        random_crop_weight = np.asarray(random_crop_weight)
                    if not FLAGS.stage_1:
                        random_crop_data_dm_list = [_dm[0,0,d_ran : d_ran+patch_size[0], 
                                                h_ran : h_ran+patch_size[1],
                                                w_ran: w_ran+patch_size[2]] for _dm in img_dm_list]

                    assert random_crop_data_t1.shape==(patch_size[0],patch_size[1],patch_size[2]), \
                            'random_crop_data shape(%s) is not in (%s,%s,%s)'%(random_crop_data_t1.shape,patch_size[0],patch_size[1],patch_size[2])

                    assert random_crop_truth.shape==(patch_size[0],patch_size[1],patch_size[2]), \
                            'random_crop_label shape is not in (%s,%s,%s)'%(patch_size[0],patch_size[1],patch_size[2])

                    
                    x1_list.append(random_crop_data_t1)
                    x2_list.append(random_crop_data_t2)
                    y_list.append(random_crop_truth)
                    if FLAGS.use_error_map:
                        weight_list.append(random_crop_weight)
                    if not FLAGS.stage_1:
                        for _i in xrange(3):
                            dm_lists[_i].append(random_crop_data_dm_list[_i])
                ##print 'A new patch is generated'
                if FLAGS.stage_1:
                    yield convert_data_multimodality(x1_list, x2_list, y_list, weight_list=weight_list)
                else:
                    yield convert_data_distancemap(x1_list, x2_list,dm_lists[0],dm_lists[1],dm_lists[2],y_list, 
                                                   weight_list=weight_list)

def get_data_list(list_file,shuffle_list=True):
    if not os.path.exists(list_file):
        print ("list_file_path: %s does not exists..." % (list_file))
        sys.exit(0)
            
    with open(list_file) as f:
        data_files = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        files_list = [x.strip() for x in data_files] 
        if FLAGS.load_with_sitk:
            files_list = [x.split(',') for x in files_list]
            
    if shuffle_list:
        shuffle(files_list)
        
    return files_list

def get_validation_split(train_list_file, validation_list_file,
                            # data_split=0.8, 
                            shuffle_list=True, 
                            overwrite_split=True):
    '''
    split the whole dataset to training and testing part
    '''
    # if overwrite_split or not os.path.exists(FLAGS.training_file):
    # print("Creating validation split...")
    training_list = get_data_list(train_list_file,shuffle_list)
    validation_list = get_data_list(validation_list_file,shuffle_list)
    
    return training_list, validation_list

def normalize_data(data, mean, std):
    data2 = data - mean
    data2 = data2 / std
    return data2


def normalize_data_storage(data_storage):
    assert len(data_storage.shape)==4, 'normalize data, must be in 4-d..'
    
    ret_data = np.zeros(data_storage.shape)
    for index in range(data_storage.shape[0]):
        _mean = data_storage[index].mean()
        _std = data_storage[index].std()
        _std = 1.0 if math.fabs(_std) < 1e-8 else _std
        ret_data[index] = normalize_data(data_storage[index], _mean, _std)
    return ret_data



def convert_test_input(test_input,normalise=True):
    test_input = np.asarray(test_input)
    
    if normalise:
        test_input = normalize_data_storage(test_input)
    
    test_input = test_input[..., np.newaxis]
    
    return test_input
    


def convert_data(train_input, train_label):
    train_input = np.asarray(train_input) 
    train_label = np.asarray(train_label) 

    train_input2 = normalize_data_storage(train_input)
    train_input2 = train_input2[..., np.newaxis]
    
    return train_input2, train_label


def convert_data_multimodality(x1_list, x2_list, y_list,weight_list=None):
    x1_list = np.asarray(x1_list)
    x2_list = np.asarray(x2_list)
    y_list = np.asarray(y_list)
    
    t1_data = normalize_data_storage(x1_list)
    t1_data = t1_data[..., np.newaxis]

    t2_data = normalize_data_storage(x2_list)
    t2_data = t2_data[..., np.newaxis]

    if weight_list is not None:
        weight_list = np.asarray(weight_list)
        return t1_data, t2_data, y_list, weight_list
    else:
        return t1_data, t2_data, y_list

def convert_data_distancemap(x1_list, x2_list,dm1_list, dm2_list,dm3_list,y_list,weight_list=None):
    dm1_list = np.asarray(dm1_list)
    dm2_list = np.asarray(dm2_list)
    dm3_list = np.asarray(dm3_list)

    dm1_data = dm1_list[..., np.newaxis]
    dm2_data = dm2_list[..., np.newaxis]
    dm3_data = dm3_list[..., np.newaxis]
    
    if weight_list is not None:
        t1_data,t2_data,y_list,weight_list = convert_data_multimodality(x1_list, x2_list, y_list, weight_list=weight_list)
        return t1_data, t2_data,dm1_data, dm2_data,dm3_data, y_list, weight_list
    else:
        t1_data,t2_data,y_list = convert_data_multimodality(x1_list, x2_list, y_list)
        return t1_data, t2_data,dm1_data, dm2_data,dm3_data, y_list

def main():
    training_generator, testing_generator = get_training_and_testing_generators()
    train_input1,  train_input2,  train_label = training_generator.next()
    print train_input1.shape

if __name__ == '__main__':
    main()
    

    