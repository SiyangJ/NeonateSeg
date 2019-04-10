from config import FLAGS
import numpy as np
import math
import h5py
from generator import convert_data,convert_test_input
from util.utils import parse_patch_size

def extract_overlapped_patches_index(img_data):
    ''' extract the start index of all overlapped patches from one image,
        img_data: depth*height*width
    '''
    if len(img_data.shape) == 3:
        img_data =  img_data[np.newaxis,np.newaxis,...]

    assert len(img_data.shape)==5, ' dimension of volume image data must be 5..'
    img_data = np.asarray(img_data, np.float32)
    
    # patch_size[] = FLAGS.patch_size[]
    patch_size = parse_patch_size(FLAGS.patch_size_str)
    overlap_add=FLAGS.overlap_add_num
    ( depth, height, width) = img_data.shape[-3:]

    d_num = depth/patch_size[0] + overlap_add
    h_num = height/patch_size[1] + overlap_add
    w_num = width/patch_size[2] + overlap_add

    d_overlap = 0 if d_num==1 else int(math.ceil(d_num*patch_size[0] - depth)  * 1.0 / (d_num -1))
    h_overlap = 0 if h_num==1 else int(math.ceil(h_num*patch_size[1] - height) * 1.0 / (h_num -1))
    w_overlap = 0 if w_num==1 else int(math.ceil(w_num*patch_size[2] - width)  * 1.0 / (w_num -1))
    # print '****  type(d_overlap) = ', type(d_overlap)
    patches_index = []
    for d in xrange(d_num):
        d_start = d*(patch_size[0] - d_overlap)
        d_end = d_start + patch_size[0]
        d_start = depth - patch_size[0]  if d_end>=depth else d_start

        for h in xrange(h_num):
            h_start = h*(patch_size[1]- h_overlap)
            h_end = h_start + patch_size[1]
            h_start = height - patch_size[1] if h_end>=height else h_start
            for w in xrange(w_num):
                w_start = w*(patch_size[2] - w_overlap)
                w_end = w_start + patch_size[2]
                w_start = width - patch_size[2]  if w_end>=width else w_start

                patches_index.append( (d_start,h_start,w_start) )

    patches_index = np.asarray(patches_index)

    return patches_index, depth, height, width

def extract_test_patches(img_data, img_label=None,normalise=True,patches_index=None):
    # patch_size[] = FLAGS.patch_size[]
    patch_size = parse_patch_size(FLAGS.patch_size_str)

    # for test, batch_size = 1
    batch_size = 1 #FLAGS.batch_size
    if img_label is not None:
        assert len(img_label.shape)==5, 'label must be in 5 dimentional..'
    if len(img_data.shape) == 3:
        img_data =  img_data[np.newaxis,np.newaxis,...]

    assert len(img_data.shape)==5, ' dimension of volume image data must be 5..'
    img_data = np.asarray(img_data, np.float32)
    if patches_index is None:
        patches_index, d, h, w = extract_overlapped_patches_index(img_data)
    else:
        d, h, w = img_data.shape[-3:]

    # print '*** patches_index:', patches_index.shape

    test_inputs = []
    test_labels = []
    test_multi_mask = []
    test_add_mask = []

    for i in xrange(patches_index.shape[0]):
        
        x_list = list()
        y_list = list()
        for _ in xrange(batch_size):
            d_ran = patches_index[i][0]
            h_ran = patches_index[i][1]
            w_ran = patches_index[i][2]

            random_crop_data = img_data[0,0,d_ran : d_ran+patch_size[0], 
                                    h_ran : h_ran+patch_size[1],
                                    w_ran: w_ran+patch_size[2]]
            random_crop_data = np.asarray(random_crop_data)
            x_list.append(random_crop_data)
            assert random_crop_data.shape==(patch_size[0],patch_size[1],patch_size[2]), \
                    'random_crop_data shape(%s) is not in (%s,%s,%s)'%(random_crop_data.shape,patch_size[0],patch_size[1],patch_size[2])
            
            if img_label is not None:
                random_crop_truth = img_label[0,0, d_ran : d_ran+patch_size[0], 
                                        h_ran : h_ran+patch_size[1],
                                        w_ran: w_ran+patch_size[2]]
                random_crop_truth = np.asarray(random_crop_truth)

                assert random_crop_truth.shape==(patch_size[0],patch_size[1],patch_size[2]), \
                        'random_crop_label shape is not in (%s,%s,%s)'%(patch_size[0],patch_size[1],patch_size[2])
                y_list.append(random_crop_truth)

        if img_label is not None:
            _input, _label = convert_data(x_list, y_list)
            test_labels.append(_label)
        else:
            _input = convert_test_input(x_list,normalise=normalise)

        test_inputs.append(_input)
        
    # file_handle.close()
    test_inputs = np.asarray(test_inputs)
    patches_index = np.asarray(patches_index)
    if img_label is not None:
        test_labels = np.asarray(test_labels)
        return (test_inputs, test_labels, patches_index, d,h,w)
        
    else:
        return (test_inputs, patches_index, d,h,w)

def main():
    
    hdf5_path = '/home/zeng/dpln/Seg3D/hdf5/0_r1f0.h5'
    patches,labels, index, d,h,w = test_patch_hdf5(hdf5_path)
    print 'patches:', patches.shape
    print 'labels:', labels.shape
    print 'index:', index.shape
    

if __name__ == '__main__':
    main()