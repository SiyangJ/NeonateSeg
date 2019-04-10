# Codes for NeonateSeg Project

Contact siyangj@live.unc.edu for questions. All rights reserved.

### Code structure
* preprocess.py  **Deprecated** Generate hdf5 file and prepare for training.
* main.py  this is the entrance where start to train
* demo.py this is where start to do test

### Test on your own data by trained model
To run test on your own data, first you need to organize test data by some rules. In default, all test data are stored in "./data_Miccai2017_ISeg/iSeg-2017-Testing" and organised like this:
>   subject-11/subject-11-T1.nii.gz
>   subject-11/subject-11-T2.nii.gz
>   ...
>   subject-20/subject-20-T1.nii.gz
>   subject-20/subject-20-T2.nii.gz

For each test sample, it should be a separate folder and the folder name  should be **subject-xx**, and inside the folder there should be T1 and T2 images which were named **subject-xx-T1.nii.gz** and **subject-xx-T2.nii.gz**, respectively.

After test data has been organised correctly,  just run the script:
```
(root)zheng@ubuntu: python demo.py
```
The segmentation result (**subject-xx_prediction.nii.gz**) and distance map(**subject-xx_cls(y)_distancemap.nii.gz**) will be saved at the same directory where the T1 and T2 data locate.  

### Training on your own data
If you want to train on your own data, there are three steps:
1. prepare training data in organisation
2. generate hdf5 file
3. start training

Before  running the script, you have to prepare all the data well. 
In default, all training data are stored at "./data_Miccai2017_ISeg/iSeg-2017-Training" and organised like this:
>   subject-1/subject-1-T1.nii.gz
>   subject-1/subject-1-T2.nii.gz
>   subject-1/subject-1-label.nii.gz
>   ...
>   subject-10/subject-10-T1.nii.gz
>   subject-10/subject-10-T2.nii.gz
>   subject-10/subject-10-label.nii.gz
>  
For each training sample, it should be a separate folder and the folder name  should be **subject-xx**, and inside the folder there should be T1, T2 and label images which were named **subject-xx-T1.nii.gz** , **subject-xx-T2.nii.gz** and **subject-xx-label.nii.gz**, respectively.

To generate hdf5 file and start to train, you only need to run the script:
```
(root)zheng@ubuntu: python preprocess.py

(root)zheng@ubuntu: python main.py
```
### Config File
All configurations can be done by setting corresponding item values  in *`config.py`*.

Some  import config items are listed as below:
1. "train_data_dir   " where stores training data
2. "test_dir"  where stores testing data
3. "checkpoint_dir" where stores trained model and logs if you start to train a new neural network
4. "last_trained_checkpoint" where stores the well-trained model. This model will be loaded if you run "demo.py" to test data
5. "restore_from_last", whether initialize weights  from a saved model when start to train a new model . If this is True, then the model stored in  "last_trained_checkpoint"  will be loaded.
6. "from_pretrain",  whether transfer learning will be used when start to train a new model . If this is True, then the weights from "hdf5_hip_transfer_model" and "hdf5_sports_3d_model" will be borrowed as the initialization value.
