from config import FLAGS
import numpy as np
import math
import h5py
import glob
from util.utils import load_nifti, save_nifti
import os



def rotate_flip(data, r=0, f_lf=False):

	#rotate 90
	data = np.rot90(data,r)

	if f_lf:
		data = np.fliplr(data)

	return data



def create_hdf5(img_data, t2_data, img_label, save_path):
	assert img_data.shape == img_label.shape, 'shape of data and label must be the same..'
	f = h5py.File(save_path, "w")
	dset = f.create_dataset("t1data", img_data.shape, dtype=np.int16)
	tset = f.create_dataset("t2data", t2_data.shape, dtype=np.int16)
	lset = f.create_dataset("label", img_data.shape, dtype=np.uint8)

	dset[...] = img_data
	lset[...] = img_label
	tset[...] = t2_data
	print('saved hdf5 file in %s' % (save_path, ))
	f.close()



def get_nifti_path():
	t1_path, t2_path, label_path = '', '', ''

	dir_list = glob.glob('%s/*/' %(FLAGS.train_data_dir,))
	# print dir_list, '....'
	for _dir in dir_list:
		# file_list = glob.glob('%s/*.nii' % (_dir, ))
		img_id = _dir.split('/')[-2]
		t1_path = '%s%s-T1.nii.gz' %(_dir, img_id)
		t2_path = '%s%s-T2.nii.gz' %(_dir, img_id)
		label_path = '%s%s-label.nii.gz' %(_dir, img_id)

		yield t1_path, t2_path, label_path
		
		


def remove_backgrounds(img_data, t2_data, img_label):
	nonzero_label = img_label != 0
	nonzero_label = np.asarray(nonzero_label)

	nonzero_index = np.nonzero(nonzero_label)
	nonzero_index = np.asarray(nonzero_index)

	x_min, x_max = nonzero_index[0,:].min(), nonzero_index[0,:].max()
	y_min, y_max = nonzero_index[1,:].min(), nonzero_index[1,:].max()
	z_min, z_max = nonzero_index[2,:].min(), nonzero_index[2,:].max()

	# print x_min, x_max
	# print y_min, y_max
	# print z_min, z_max

	x_min = x_min - FLAGS.prepost_pad if x_min-FLAGS.prepost_pad>=0 else 0
	y_min = y_min - FLAGS.prepost_pad if y_min-FLAGS.prepost_pad>=0 else 0
	z_min = z_min - FLAGS.prepost_pad if z_min-FLAGS.prepost_pad>=0 else 0

	x_max = x_max + FLAGS.prepost_pad if x_max+FLAGS.prepost_pad<=img_data.shape[0] else img_data.shape[0]
	y_max = y_max + FLAGS.prepost_pad if y_max+FLAGS.prepost_pad<=img_data.shape[1] else img_data.shape[1]
	z_max = z_max + FLAGS.prepost_pad if z_max+FLAGS.prepost_pad<=img_data.shape[2] else img_data.shape[2]


	return (img_data[x_min:x_max, y_min:y_max, z_min:z_max], t2_data[x_min:x_max, y_min:y_max, z_min:z_max],
					img_label[x_min:x_max, y_min:y_max, z_min:z_max])




def generate_nifti_data():

	for img_path, t2_path, label_path in get_nifti_path():
		print img_path, t2_path, label_path
		nifti_data, nifti_img = load_nifti(img_path)
		t2_data, t2_img = load_nifti(t2_path)
		nifti_label, _label = load_nifti(label_path)
		print '>> img_path=', img_path
		img_id = img_path.split('/')[-2]
		print '>> img_id=', img_id

		if len(nifti_data.shape)==3:
			pass
		elif len(nifti_data.shape)==4:
			nifti_data = nifti_data[:,:,:,0]
			t2_data = t2_data[:,:,:,0]
			nifti_label = nifti_label[:,:,:,0]
		

		t1_data = np.asarray(nifti_data, np.int16)
		t2_data = np.asarray(t2_data, np.int16)
		
		nifti_label = np.asarray(nifti_label, np.uint8)

		nifti_label[nifti_label==10] = 1
		nifti_label[nifti_label==150] = 2
		nifti_label[nifti_label==250] = 3

		print '*** t1_data.shape=', t1_data.shape
		print '*** nifti_label.shape=', nifti_label.shape
		##croped_data, t2_data, croped_label = remove_backgrounds(t1_data,t2_data, nifti_label)
		croped_data, t2_data, croped_label = (t1_data,t2_data, nifti_label)

		t1_name = img_path.split('/')[-1].replace('.nii.gz', '')
		t2_name = t2_path.split('/')[-1].replace('.nii.gz', '')
		
		
		for _r in xrange(4):
			for flip in [True, False]:
				save_path = '%s/%s_r%d_f%d.h5' %(FLAGS.hdf5_dir, img_id, _r, flip)
				print ('>> start to creat hdf5: %s' % (save_path,))
				aug_data = rotate_flip(croped_data, r=_r, f_lf=flip )
				aug_label = rotate_flip(croped_label, r=_r, f_lf=flip )
				aug_t2_data = rotate_flip(t2_data,  r=_r, f_lf=flip)
				
				create_hdf5(aug_data,aug_t2_data, aug_label, save_path)

				save_nifit_path = '%s/%s_r%d_f%d_data.nii' % (FLAGS.hdf5_dir, t1_name,_r, flip )
				save_nifit_label_path = '%s/%s_r%d_f%d_label.nii' % (FLAGS.hdf5_dir, img_id, _r, flip)
				t2_path = '%s/%s_r%d_f%d_data.nii' % (FLAGS.hdf5_dir, t2_name, _r, flip)
				print '>>.. save_nifit_path=', save_nifit_path
				print '>>.. save_nifit_label_path=', save_nifit_label_path
				print '>>.. t2_path=', t2_path

				print ''

		# break




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