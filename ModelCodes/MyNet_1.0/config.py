import os
import sys
import tensorflow as tf
import configparser
FLAGS = tf.app.flags.FLAGS

## Configuration File Parse
CONFIG_DIR = './config.ini'
if len(sys.argv)>1 and sys.argv[1][-4:]=='.ini':
    CONFIG_DIR = sys.argv[1]
print('Using {:s} as config file.'.format(CONFIG_DIR))
CFP = configparser.ConfigParser()
CFP.read(CONFIG_DIR)

ARGS = CFP['Default']

tf.app.flags.DEFINE_integer('cls_out', ARGS.getint('cls_out'), 
                            "classfy how many categories")
tf.app.flags.DEFINE_integer('batch_size', ARGS.getint('batch_size'), 
                            "Number of samples per batch.")
tf.app.flags.DEFINE_integer('accumulate_times', ARGS.getint('accumulate_times',1), 
                            "Accumulate the gradients to make effectively larger batch size.")
tf.app.flags.DEFINE_integer('val_accumulate_times', ARGS.getint('val_accumulate_times',ARGS.getint('accumulate_times')), 
                            "Accumulate the gradients to make effectively larger batch size.")

tf.app.flags.DEFINE_bool('calculate_class_weights', ARGS.getboolean('calculate_class_weights',False), 
                         "whether to calculate class weights to balance the training")
tf.app.flags.DEFINE_string('class_weights_string', ARGS.get('provide_class_weights',"1.0,1.0,1.0,1.0"), 
                           "Class weights used to balance the labels; default is 1,1,1,1.")

tf.app.flags.DEFINE_integer('validate_every_n', ARGS.getint('validate_every_n',10), 
                            "Validate the training every n steps.")

tf.app.flags.DEFINE_integer('early_stop_iteration', 
                            ARGS.getint('early_stop_iteration',FLAGS.batch_size*FLAGS.val_accumulate_times*100), 
                            "Stop the training and go back to the last best result.")

tf.app.flags.DEFINE_integer('early_stop_max_fail', 
                            ARGS.getint('early_stop_max_fail',3), 
                            "Stop the training and go back to the last best result.")

tf.app.flags.DEFINE_string('patch_size_str', ARGS['patch_size_str'], 
                           "patch size that we will extract from 3D image")
tf.app.flags.DEFINE_integer('batches_one_image', ARGS.getint('batches_one_image'), 
                            "how many batches extraction from a 3D image for training")
tf.app.flags.DEFINE_integer('overlap_add_num', ARGS.getint('overlap_add_num'), 
                            "patch_num = Len/patch_size + overlap_add_num when extracting patches for test")
tf.app.flags.DEFINE_integer('prepost_pad', ARGS.getint('prepost_pad'), 
                            "padding when remove zero backgrounds in preprocess")
tf.app.flags.DEFINE_integer('training_crop_pad', ARGS.getint('training_crop_pad'), 
                            "padding when remove zero backgrounds in preprocess")

tf.app.flags.DEFINE_bool('load_with_sitk', ARGS.getboolean('load_with_sitk',False),
                         "whether to directly load images with Simple ITK")
tf.app.flags.DEFINE_bool('preload_data', ARGS.getboolean('preload_data',True), 
                         "whether to load the data in memory for efficiency; useful when data augmentation is enabled")
## TODO
## Currently, data augmentation is only supported by load_sitk and pre_load

tf.app.flags.DEFINE_bool('augmentation', ARGS.getboolean('augmentation',False), 
                         "whether to use data augmentation to regularize the network")
assert (not FLAGS.augmentation) or (FLAGS.preload_data and FLAGS.load_with_sitk), \
    "Currently, data augmentation is only supported by load_sitk and pre_load"

if FLAGS.augmentation:
    tf.app.flags.DEFINE_string('rotation_degree', ARGS.get('rotation_degree',None), 
                           "Rotation in degrees in data augmentation")
    tf.app.flags.DEFINE_string('flip', ARGS.get('flip',None), 
                           "Flip in data augmentation")
    tf.app.flags.DEFINE_bool('deformation', ARGS.getboolean('deformation',False), 
                           "Whether to use elastic deformation in data augmentation")
    
    if FLAGS.deformation:
        tf.app.flags.DEFINE_integer('deformation_sigma', ARGS.getint('deformation_sigma',5), 
                           "Deformation sigma in BSpline")
        tf.app.flags.DEFINE_integer('num_control_points', ARGS.getint('num_control_points',5), 
                           "Number of control points in BSpline deformation")
    
    tf.app.flags.DEFINE_string('scaling_percentage', ARGS.get('scaling_percentage',None), 
                           "Scaling in percentage in data augmentation")
    
    '''
    rotation_degree = 5,5,5
    flip = 1,0,0
    deformation = True
    deformation_sigma = 5
    num_control_points = 5
    scaling_percentage = 5,5,5
    '''

tf.app.flags.DEFINE_integer('augmentation_per_n_epoch', ARGS.getint('augmentation_per_n_epoch',3), 
                            "Augmentation is done per n epoch for efficiency")

############### Training and Learning rate decay ##################################
tf.app.flags.DEFINE_float('momentum', ARGS.getfloat('momentum'), 
                          "momentum for accelearating training")
tf.app.flags.DEFINE_float('learning_rate_start', ARGS.getfloat('learning_rate_start'), 
                          "start learning rate ")
tf.app.flags.DEFINE_integer('learning_rate_reduce_life', ARGS.getint('learning_rate_reduce_life'), 
                            "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_float('learning_rate_percentage', ARGS.getfloat('learning_rate_percentage'), 
                          "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_integer('max_batch', ARGS.getint('max_batch'),
                            "max batch number")
tf.app.flags.DEFINE_integer('checkpoint_period', ARGS.getint('checkpoint_period'), 
                            "Number of batches in between checkpoints")
tf.app.flags.DEFINE_string('checkpoint_dir', ARGS['checkpoint_dir'], 
                           "Output folder where training logs and models are dumped.")
#tf.app.flags.DEFINE_string('last_trained_checkpoint', './checkpoint_t1_t2_9case_10000', "The model used for testing..")
tf.app.flags.DEFINE_string('last_trained_checkpoint', ARGS['last_trained_checkpoint'], 
                           "The model used for testing")
tf.app.flags.DEFINE_bool('restore_from_last', ARGS.getboolean('restore_from_last'), 
                         "whether start training from last trained checkpoint")

############### Deep supervision######################
tf.app.flags.DEFINE_float('aux1_weight', ARGS.getfloat('aux1_weight'), 
                          "loss weight of aux1 classifier")
tf.app.flags.DEFINE_float('aux2_weight', ARGS.getfloat('aux2_weight'), 
                          "loss weight of aux2 classifier")
tf.app.flags.DEFINE_float('main_weight', ARGS.getfloat('main_weight'), 
                          "loss weight of main classifier")
tf.app.flags.DEFINE_float('L2_loss_weight', ARGS.getfloat('L2_loss_weight'), 
                          "loss weight of main classifier")
# tf.app.flags.DEFINE_float('reject_T', 0.05, "remove isolated regions, when the area is less then reject_T")



################### Train Data################

tf.app.flags.DEFINE_bool('overwrite_split', ARGS.getboolean('overwrite_split',False),
                         "whether to overwrite existing data splitting file")

tf.app.flags.DEFINE_string('hdf5_dir', ARGS['hdf5_dir'],
                           "Store the path which contains hdf5 files.")
tf.app.flags.DEFINE_string('train_data_dir', ARGS['train_data_dir'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_list_path', ARGS['hdf5_list_path'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_train_list_path', ARGS['hdf5_train_list_path'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_validation_list_path', ARGS['hdf5_validation_list_path'], 
                           "Store the validation hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_test_list_path', ARGS.get('hdf5_test_list_path',None), 
                           "Store the test hdf5 file list.")

tf.app.flags.DEFINE_string('metric_used_on_test', ARGS.get('metric_used_on_test','same'), 
                           "Metric used for testing.")
tf.app.flags.DEFINE_bool('early_stop_on_test', ARGS.getboolean('early_stop_on_test',False), 
                         "whether early stop on test, default is False")

################# Pretrain Model: Partial Transfer Learning  ########################################################
tf.app.flags.DEFINE_bool('from_pretrain', ARGS.getboolean('from_pretrain'), 
                         "when init value from pretrain-ed model")
tf.app.flags.DEFINE_string('hdf5_hip_transfer_model', ARGS['hdf5_hip_transfer_model'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_string('hdf5_sports_3d_model', ARGS['hdf5_sports_3d_model'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_string('model_saved_hdf5', ARGS['model_saved_hdf5'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_bool('xavier_init', ARGS.getboolean('xavier_init'),
                         "whether multi-modality is used")


tf.app.flags.DEFINE_bool('log_device_placement', ARGS.getboolean('log_device_placement'), 
                         "Log the device where variables are placed.")
tf.app.flags.DEFINE_integer('random_seed', ARGS.getint('random_seed'), 
                            "Seed used to initialize rng.")
tf.app.flags.DEFINE_float('epsilon', ARGS.getfloat('epsilon'), 
                          "Fuzz term to avoid numerical instability")


################ Test Data ###############################
#tf.app.flags.DEFINE_string('test_dir','/proj/NIRAL/users/jphong/6moSegData/IBIS/Test',"the directory which contains nifti images to be segmented.")
tf.app.flags.DEFINE_string('test_dir', ARGS['test_dir'],
                           "the directory which contains nifti images to be segmented.")
tf.app.flags.DEFINE_bool('load_test_with_sitk', ARGS.getboolean('load_test_with_sitk',True), 
                         "load the test/inference images with SimpleITK.")
tf.app.flags.DEFINE_string('prediction_save_dir', ARGS.get('prediction_save_dir',FLAGS.checkpoint_dir),
                           "The directory to save the predictions.")
'''
load_test_with_sitk=True
prediction_save_dir=/proj/NIRAL/users/siyangj/NewModels/model_0217_data_aug/models/1/data_aug1
'''

def main():
    print FLAGS.testing_file

if __name__ == '__main__':
    main()
