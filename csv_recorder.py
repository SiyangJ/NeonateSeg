import pandas as pd
import numpy as np
import re

def find_val(sec,attr,s):
    r = re.compile(r'(?:\['+sec+r'\](?:.*?)-- '+attr+r': )(.*?)(?:\n)',re.S)
    m = r.search(s)
    if m is None:
        return None
    else:
        return m.group(1)

ini_dict = {"CONFIG_FILE":["path",],
            "PARCELLATION":["loader","csv_file","axcodes","filename_contains","filename_not_contains","interp_order","pixdim","spatial_window_size","path_to_search",],
            "T1":["loader","csv_file","axcodes","filename_contains","filename_not_contains","interp_order","pixdim","spatial_window_size","path_to_search",],
            "CUSTOM":["num_classes","compulsory_labels","output_prob","label_normalisation","name","inferred","weight","min_sampling_ratio","sampler","image","proba_connect","min_numb_labels","rand_samples","label","softmax","evaluation_units",],
            "T2":["loader","csv_file","axcodes","filename_contains","filename_not_contains","interp_order","pixdim","spatial_window_size","path_to_search",],
            "EVALUATION":["evaluations","save_csv_dir",],"INFERENCE":["dataset_to_infer","output_interp_order","border","save_seg_dir","output_postfix","inference_iter","spatial_window_size",],
            "TRAINING":["deformation_sigma","validation_every_n","loss_type","max_checkpoints","starting_iter","proportion_to_deform","do_elastic_deformation","exclude_fraction_for_validation","sample_per_volume","scaling_percentage","max_iter","save_every_n","rotation_angle_z","validation_max_iter","random_flipping_axes","lr","optimiser","num_ctrl_points","exclude_fraction_for_inference","rotation_angle_x","rotation_angle_y","tensorboard_every_n","rotation_angle",],"SYSTEM":["num_threads","event_handler","dataset_split_file","loader","model_dir","cuda_devices","action","num_gpus","iteration_generator",],"NETWORK":["keep_prob","foreground_type","volume_padding_size","reg_type","queue_length","decay","activation_function","histogram_ref_file","volume_padding_mode","cutoff","whitening","multimod_foreground_type","batch_size","weight_initializer","window_sampling","name","normalise_foreground_only","bias_initializer","normalisation","norm_type"]}

def record_ini(f,c):
    df = pd.read_csv(c)
    ifile = open(f,'rb+')
    istr = ifile.read().decode('utf-8')
    l = len(df.index)
    p = find_val('CONFIG_FILE','path',istr)
    if len(df.loc[df['CONFIG_FILE-path']==p])>0:
        return
    df.loc[l]=0
    for sec,attr_list in ini_dict.items():
        for attr in attr_list:
            v = find_val(sec,attr,istr)
            #print(sec,attr,v)
            if v is not None:
                df.loc[l,sec+'-'+attr]=v
    df.to_csv(c,index=False)
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('ini_path', type=str)
    args = parser.parse_args()
    record_ini(args.ini_path,"model_record.csv")