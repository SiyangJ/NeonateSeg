from config import FLAGS
import numpy as np
import math
import h5py
import glob
from util.utils import load_nifti, save_nifti
# from postprocess import remove_isolated_minor
import os



def rotate_flip(data, r=0, f_lf=False):

	#rotate 90
	data = np.rot90(data,r)

	if f_lf:
		data = np.fliplr(data)

	return data



def create_hdf5(t1_data, t2_data, img_label, dm1_data, dm2_data, dm3_data, save_path):
	'''
		dm1_data, dm2_data, dm3_data must be normalised.... divided by maxmium value 
	'''
	assert t1_data.shape == img_label.shape, 'shape of data and label must be the same..'
	f = h5py.File(save_path, "w")
	dset = f.create_dataset("t1data", t1_data.shape, dtype=np.int16)
	tset = f.create_dataset("t2data", t2_data.shape, dtype=np.int16)
	lset = f.create_dataset("label", t1_data.shape, dtype=np.uint8)
	
	dm1set = f.create_dataset("dm1", t1_data.shape, dtype=np.float)
	dm2set = f.create_dataset("dm2", t1_data.shape, dtype=np.float)
	dm3set = f.create_dataset("dm3", t1_data.shape, dtype=np.float)


	dset[...] = t1_data 
	lset[...] = img_label
	tset[...] = t2_data

	dm1set[...] = dm1_data
	dm2set[...] = dm2_data 
	dm3set[...] = dm3_data

	print('saved hdf5 file in %s' % (save_path, ))
	f.close()


def get_nifti_path():
	t1_path, t2_path, label_path, dm1_path, dm2_path, dm3_path = '', '', '','', '',''


	dir_list = glob.glob('%s/*/' %(FLAGS.nifti_dir,))
	# print dir_list, '....'
	for _dir in dir_list:
		# file_list = glob.glob('%s/*.nii' % (_dir, ))
		img_id = _dir.split('/')[-2]
		print '>> '
		t1_path = '%s%s-T1.nii.gz' %(_dir, img_id)
		t2_path = '%s%s-T2.nii.gz' %(_dir, img_id)
		label_path = '%s%s-label.nii.gz' %(_dir, img_id)

		dm1_path = '%s%s_cls1_distancemap.nii.gz' % (_dir, img_id)
		dm2_path = '%s%s_cls1_distancemap.nii.gz' % (_dir, img_id)
		dm3_path = '%s%s_cls1_distancemap.nii.gz' % (_dir, img_id)

		yield t1_path, t2_path, label_path, dm1_path, dm2_path, dm3_path
		

def remove_training_distance_map_backgrounds(img_data, t2_data, label_data, dm1_data, dm2_data, dm3_data):
	nonzero_label = img_data != 0
	nonzero_label = np.asarray(nonzero_label)
	

	nonzero_index = np.nonzero(nonzero_label)
	nonzero_index = np.asarray(nonzero_index)

	x_min, x_max = nonzero_index[0,:].min(), nonzero_index[0,:].max()
	y_min, y_max = nonzero_index[1,:].min(), nonzero_index[1,:].max()
	z_min, z_max = nonzero_index[2,:].min(), nonzero_index[2,:].max()


	x_min = x_min - FLAGS.prepost_pad if x_min-FLAGS.prepost_pad>=0 else 0
	y_min = y_min - FLAGS.prepost_pad if y_min-FLAGS.prepost_pad>=0 else 0
	z_min = z_min - FLAGS.prepost_pad if z_min-FLAGS.prepost_pad>=0 else 0

	x_max = x_max + FLAGS.prepost_pad if x_max+FLAGS.prepost_pad<=img_data.shape[0] else img_data.shape[0]
	y_max = y_max + FLAGS.prepost_pad if y_max+FLAGS.prepost_pad<=img_data.shape[1] else img_data.shape[1]
	z_max = z_max + FLAGS.prepost_pad if z_max+FLAGS.prepost_pad<=img_data.shape[2] else img_data.shape[2]

	crop_index = (x_min,x_max, y_min, y_max, z_min, z_max)

	return (img_data[x_min:x_max, y_min:y_max, z_min:z_max],
			t2_data[x_min:x_max, y_min:y_max, z_min:z_max],
			label_data[x_min:x_max, y_min:y_max, z_min:z_max],
				dm1_data[x_min:x_max, y_min:y_max, z_min:z_max],
				dm2_data[x_min:x_max, y_min:y_max, z_min:z_max],
				dm3_data[x_min:x_max, y_min:y_max, z_min:z_max],crop_index)



def generate_nifti_data():


	for t1_path, t2_path, label_path, dm1_path, dm2_path, dm3_path in get_nifti_path():
		# print img_path, t2_path, label_path
		t1_data, t1_img = load_nifti(t1_path)
		t2_data, t2_img = load_nifti(t2_path)
		img_label, _label = load_nifti(label_path)
		dm1_data, dm1_img = load_nifti(dm1_path)
		dm2_data, dm2_img = load_nifti(dm2_path)
		dm3_data, dm3_img = load_nifti(dm3_path)

		
		nifti_name = t1_path.split('/')[-2]
		dm1_data /= np.max(dm1_data)
		dm2_data /= np.max(dm2_data)
		dm3_data /= np.max(dm3_data)
		
		if len(t1_data.shape)==3:
			pass
		elif len(t1_data.shape)==4:
			t1_data = t1_data[:,:,:,0]
			t2_data = t2_data[:,:,:,0]
			img_label = img_label[:,:,:,0]


		img_label[img_label==10] = 1
		img_label[img_label==150] = 2
		img_label[img_label==250] = 3
		img_label = np.asarray(img_label, np.uint8)
		print '>> dm3_data=',dm3_data.shape

		t1_data = np.asarray(t1_data, np.int16)
		t2_data = np.asarray(t2_data, np.int16)
		img_label = np.asarray(img_label, np.int16)
		dm1_data = np.asarray(dm1_data, np.float)
		dm2_data = np.asarray(dm2_data, np.float)
		dm3_data = np.asarray(dm3_data, np.float)
		
		t1_data, t2_data,img_label, dm1_data, dm2_data, dm3_data,crop_index = \
								remove_training_distance_map_backgrounds(t1_data, t2_data, img_label,dm1_data, dm2_data, dm3_data)
		
		for _r in xrange(4):
			for flip in [ False, True]:
				save_path = '%s/%s_r%d_f%d.h5' %(FLAGS.hdf5_dir, nifti_name, _r, flip)
				print ('>> start to creat hdf5: %s' % (save_path,))
				
				
				create_hdf5(t1_data, t2_data, img_label, dm1_data, dm2_data, dm3_data, save_path)

				save_nifit_path = '%s/%s_r%d_f%d_t1_data.nii.gz' % (FLAGS.hdf5_dir, nifti_name,_r, flip )
				save_nifit_label_path = '%s/%s_r%d_f%d_label.nii.gz' % (FLAGS.hdf5_dir, nifti_name, _r, flip)
				t2_path = '%s/%s_r%d_f%d_t2_data.nii.gz' % (FLAGS.hdf5_dir, nifti_name, _r, flip)
				



def generate_file_list():
	# if os.pa
	file_list = glob.glob('%s/*.h5' %(FLAGS.hdf5_dir,))
	file_list.sort()
	with open(FLAGS.hdf5_list_path, 'w') as _file:
		for _file_path in file_list:
			_file.write(_file_path)
			_file.write('\n')

	print '>> finish writing to file:', FLAGS.hdf5_list_path

	with open(FLAGS.hdf5_train_list_path, 'w') as _file:
		for _file_path in file_list[8:]:
			_file.write(_file_path)
			_file.write('\n')

	print '>> finish writing to file:', FLAGS.hdf5_train_list_path

	with open(FLAGS.hdf5_validation_list_path, 'w') as _file:
		for _file_path in file_list[0:8]:
			_file.write(_file_path)
			_file.write('\n')
	print '>> finish writing to file:', FLAGS.hdf5_validation_list_path
	

def main():
	
	generate_nifti_data()
	generate_file_list()
	
	

	

if __name__ == '__main__':
	main()