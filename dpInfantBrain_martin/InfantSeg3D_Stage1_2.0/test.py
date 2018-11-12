from config import FLAGS
import numpy as np
import math
import h5py
import glob
from util.utils import load_nifti, save_nifti
import os

def get_nifti_path():
	t1_path, t2_path, label_path = '', '', ''

	dir_list = glob.glob('%s/*/' %(FLAGS.test_dir,))
	print '>> test_dir=', FLAGS.test_dir
	
	for _dir in dir_list:
		img_id = _dir.split('/')[-2]
		t1_path = '%s%s-T1.img' %(_dir, img_id)
		t2_path = '%s%s-T2.img' %(_dir, img_id)
		

		yield t1_path, t2_path#, label_path


def ensure_dir(file_path):
	import os
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

def generate_nifti_data():

	save_train_dir = './data_Miccai2017_ISeg/test_data'
	# for img_path, t2_path, label_path in get_nifti_path():
	for img_path, t2_path in get_nifti_path():
		print img_path, t2_path
		nifti_data, nifti_img = load_nifti(img_path)
		t2_data, t2_img = load_nifti(t2_path)
		# nifti_label, _label = load_nifti(label_path)
		print '>> img_path=', img_path
		img_id = img_path.split('/')[-2]
		print '>> img_id=', img_id

		save_img_path = img_path.replace(FLAGS.test_dir, save_train_dir).replace('.img','.nii.gz')
		save_t2_path = t2_path.replace(FLAGS.test_dir, save_train_dir).replace('.img','.nii.gz')

		print '>> save_img_path=',save_img_path
		print '>> save_t2_path=',save_t2_path
		
		ensure_dir(save_img_path)
		save_nifti(nifti_data.astype(np.int16), nifti_img.affine, save_img_path)
		save_nifti(t2_data.astype(np.int16), t2_img.affine, save_t2_path)
		



def main():
	generate_nifti_data()
	pass

if __name__ == '__main__':
	main()