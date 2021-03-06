import pickle
import nibabel as nib
import SimpleITK as sitk
import numpy as np

def Dice(pred,true,labels=[0,1,2,3]):
    l = len(labels)
    ret = np.zeros(l)
    for i in range(l):
        lab = labels[i]
        p = pred==lab
        t = true==lab
        ret[i] = 2*float(np.logical_and(p,t).sum())/(p.sum()+t.sum())
    return ret


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def load_sitk(sitk_path):
    return np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(str(sitk_path))),0,2)

def load_nifti(nifti_path):
    img1 = nib.load(nifti_path)
    data = img1.get_data()
    affine = img1.affine
    data = np.asarray(data)
    return data, img1


def save_nifti(img_data, affine, save_path):
    new_image = nib.Nifti1Image(img_data, affine)
    nib.save(new_image, save_path)
    
def save_sitk(img_data,save_path):
    '''
    print(save_path)
    print(type(save_path))
    print(type(str(save_path)))
    '''
    sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(img_data,0,2)),str(save_path))

def save_hdr_img(img_data, affine, header, save_path):
    # import nibabel as nib
    #how to get affine and header
    print '** >> save_hdr_img ', img_data.shape, img_data.dtype
    img = nib.Nifti1Image(img_data, affine, header)
    img.set_data_dtype(np.uint8)
    # img.set_data_dtype(np.float)
    nib.nifti1.save(img, save_path)

def parse_string_to_numbers(pat_str,to_type=float):
    arr_str = pat_str.split(',')
    return tuple(to_type(s) for s in arr_str)
    
def parse_patch_size(pat_str):
    arr_str = pat_str.split(',')
    return (int(arr_str[0]),int(arr_str[1]),int(arr_str[2])) 

def parse_class_weights(pat_str):
    arr_str = pat_str.split(',')
    return (float(arr_str[0]),float(arr_str[1]),float(arr_str[2]),float(arr_str[3])) 

def main():
    pat_str = '32,64,128'
    xx = parse_patch_size(pat_str)
    print xx[0]

if __name__ == '__main__':
    main()