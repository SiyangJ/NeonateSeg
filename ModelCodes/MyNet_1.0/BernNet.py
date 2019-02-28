import tensorflow as tf
import numpy as np
from pretrain_transfer import get_pretrained_weights
from util.utils import parse_patch_size,parse_class_weights

from config import FLAGS

from layers import _weight_variable, _bias_variable, cal_loss, add_conv3d, add_deconv3d, add_maxpool3d, _get_var_list

DTYPE = tf.float32

def create_model_infant_seg(train_phase=True):
    
    from_pretrain = False if not train_phase else FLAGS.from_pretrain
    num_example = FLAGS.batch_size if train_phase else 1

    patch_size = parse_patch_size(FLAGS.patch_size_str)

    channels = 1
    input_shape = (num_example, patch_size[0], patch_size[1], patch_size[2], channels)
    label_shape = (num_example,  patch_size[0], patch_size[1], patch_size[2])
    label_shape1 = (num_example,  patch_size[0]/4, patch_size[1]/4, patch_size[2]/4)
    label_shape2 = (num_example,  patch_size[0]/2, patch_size[1]/2, patch_size[2]/2)
    mask_shape = (num_example, patch_size[0], patch_size[1], patch_size[2], FLAGS.cls_out )

    old_vars = tf.global_variables()
    
    tf_t1_input = tf.placeholder(tf.float32, shape=input_shape)
    tf_t2_input = tf.placeholder(tf.float32, shape=input_shape)
    tf_label_main = tf.placeholder(tf.int32, shape=label_shape)

    # tf_label_main = tf.to_float(tf_label_main)
    tf_label_aux1 = add_maxpool3d('aux1_label', tf.expand_dims(tf.to_float(tf_label_main), -1), stride=4)
    tf_label_aux1 = tf.to_int32(tf.reshape(tf_label_aux1, label_shape1))
    tf_label_aux2 = add_maxpool3d('aux2_label', tf.expand_dims(tf.to_float(tf_label_main), -1), stride=2)
    tf_label_aux2 = tf.to_int32(tf.reshape(tf_label_aux2, label_shape2))
    
    in_filters = tf_t1_input.shape[-1]
    conv1_filters = 64
    t1_conv1a= add_conv3d('t1_conv1a', tf_t1_input, in_filters, conv1_filters, 
                                                train_phase=train_phase)

    t1_pool1 = add_maxpool3d('t1_pool1', t1_conv1a, stride=2)

    # conv2 
    conv2_filters = 128
    t1_conv2a = add_conv3d('t1_conv2a', t1_pool1, conv1_filters, conv2_filters,
                                                train_phase=train_phase )
    t1_pool2 = add_maxpool3d('t1_pool2', t1_conv2a, stride=2)

    # conv3(a+b) 
    conv3_filters = 256
    t1_conv3a = add_conv3d('t1_conv3a', t1_pool2, conv2_filters, conv3_filters,
                                                train_phase=train_phase)
    t1_conv3b = add_conv3d('t1_conv3b', t1_conv3a, conv3_filters, conv3_filters, 
                                                train_phase=train_phase)
    t1_pool3 = add_maxpool3d('t1_pool3', t1_conv3b, stride=2)

    #conv4(a+b)
    conv4_filters = 512
    t1_conv4a = add_conv3d('t1_conv4a', t1_pool3, conv3_filters, conv4_filters, 
                                                train_phase=train_phase)
    t1_conv4b = add_conv3d('t1_conv4b', t1_conv4a, conv4_filters, conv4_filters,
                                                train_phase=train_phase)


    ###################### add t2-modality ############################

    in_filters = tf_t2_input.shape[-1]
    conv1_filters = 64
    t2_conv1a= add_conv3d('t2_conv1a', tf_t2_input, in_filters, conv1_filters, 
                                                train_phase=train_phase)

    t2_pool1 = add_maxpool3d('t2_pool1', t2_conv1a, stride=2)

    # conv2 
    conv2_filters = 128
    t2_conv2a = add_conv3d('t2_conv2a', t2_pool1, conv1_filters, conv2_filters,
                                                train_phase=train_phase )
    t2_pool2 = add_maxpool3d('t2_pool2', t2_conv2a, stride=2)

    # conv3(a+b) 
    conv3_filters = 256
    t2_conv3a = add_conv3d('t2_conv3a', t2_pool2, conv2_filters, conv3_filters,
                                                train_phase=train_phase)
    t2_conv3b = add_conv3d('t2_conv3b', t2_conv3a, conv3_filters, conv3_filters, 
                                                train_phase=train_phase)
    t2_pool3 = add_maxpool3d('t2_pool3', t2_conv3b, stride=2)

    #conv4(a+b)
    conv4_filters = 512
    t2_conv4a = add_conv3d('t2_conv4a', t2_pool3, conv3_filters, conv4_filters, 
                                                train_phase=train_phase)
    t2_conv4b = add_conv3d('t2_conv4b', t2_conv4a, conv4_filters, conv4_filters,
                                                train_phase=train_phase)

    conv4b = tf.concat([t1_conv4b, t2_conv4b], -1)

    ###################################################################


    # deconv1(a+b+c)
    deconv1_filters = 256
    deconv1a = add_deconv3d('deconv1a', conv4b, conv4_filters+conv4_filters, deconv1_filters, 
                                                stride=2 , train_phase=train_phase)
    # concat
    # print ('** conv3b ', conv3b.shape, conv3b.dtype)
    concat1 = tf.concat([deconv1a, t1_conv3b], -1)
    # numpy.concatenate((a1, a2, ...), axis=0)
    deconv1b = add_conv3d('deconv1b', concat1, deconv1_filters*2, deconv1_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv1c = add_conv3d('deconv1c', deconv1b, deconv1_filters, deconv1_filters,
                                                filter_size=3,  train_phase=train_phase)

    # deconv2(a+b+c)
    deconv2_filters = 128
    deconv2a = add_deconv3d('deconv2a', deconv1c, deconv1_filters, deconv2_filters, 
                                                stride=2 , train_phase=train_phase)
    concat2 = tf.concat( [deconv2a, t1_conv2a], -1)
    deconv2b = add_conv3d('deconv2b', concat2, deconv2_filters*2, deconv2_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv2c = add_conv3d('deconv2c', deconv2b,deconv2_filters, deconv2_filters,
                                                filter_size=3,  train_phase=train_phase)



    # deconv3(a+b+c)
    deconv3_filters = 64
    deconv3a = add_deconv3d('deconv3a', deconv2c, deconv2_filters, deconv3_filters, #32?
                                                stride=2 , train_phase=train_phase)
    concat3 = tf.concat( [deconv3a, t1_conv1a], -1)
    deconv3b = add_conv3d('deconv3b', concat3, deconv3_filters*2, deconv3_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv3c = add_conv3d('deconv3c', deconv3b, deconv3_filters, deconv3_filters,
                                                filter_size=3,  train_phase=train_phase)

    #output
    # last layer before output, should not contain relu layer
    final_out_filters = FLAGS.cls_out

    #aux1 output 
    aux1_conv = add_conv3d('aux1_conv', deconv1c, deconv1_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)
    aux1_pred = add_conv3d('aux1_pred', aux1_conv, final_out_filters, final_out_filters,
                                                stride=1, train_phase=train_phase,
                                                with_batch_relu=False)

    #aux2 output 
    aux2_conv = add_conv3d('aux2_conv', deconv2c, deconv2_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)
    aux2_pred = add_conv3d('aux2_pred', aux2_conv, final_out_filters, final_out_filters,
                                                stride=1, train_phase=train_phase,
                                                with_batch_relu=False)

    #main output
    main_pred = add_conv3d('main_pred', deconv3c, deconv3_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)

    main_possibility = tf.nn.softmax(main_pred)



    new_vars = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))




    aux1_loss = cal_loss(aux1_pred, tf_label_aux1)
    aux2_loss = cal_loss(aux2_pred, tf_label_aux2)
    main_loss = cal_loss(main_pred, tf_label_main)


    final_loss = aux1_loss*FLAGS.aux1_weight + aux2_loss*FLAGS.aux2_weight \
                                             + main_loss*FLAGS.main_weight



    # l2 loss
    # for each varibale name, like:  conv2a/biases:0  and aux2_pred/weights:0
    for _var in gene_vars:
       
        # to use L2 loss, all the variables must be with the name of "weights"
        if _var.name[-9:-2] == 'weights': # to exclude 'bias' term
            final_loss = final_loss + FLAGS.L2_loss_weight*tf.nn.l2_loss(_var)
        
    
    return (tf_t1_input, tf_t2_input, tf_label_main, 
            aux1_pred, aux2_pred, main_pred,
            aux1_loss, aux2_loss, main_loss, 
            final_loss, gene_vars, main_possibility)

def create_model_infant_t1t2dm123_seg(train_phase=True):
    
    from_pretrain = False if not train_phase else FLAGS.from_pretrain
    num_example = FLAGS.batch_size if train_phase else 1

    # patch_size_str = 

    patch_size = parse_patch_size(FLAGS.patch_size_str)

    channels = 1
    input_shape = (num_example, patch_size[0], patch_size[1], patch_size[2], channels)
    label_shape = (num_example,  patch_size[0], patch_size[1], patch_size[2])
    label_shape1 = (num_example,  patch_size[0]/4, patch_size[1]/4, patch_size[2]/4)
    label_shape2 = (num_example,  patch_size[0]/2, patch_size[1]/2, patch_size[2]/2)
    mask_shape = (num_example, patch_size[0], patch_size[1], patch_size[2], FLAGS.cls_out )

    old_vars = tf.global_variables()
   
    
    tf_t1_input = tf.placeholder(tf.float32, shape=input_shape)
    tf_t2_input = tf.placeholder(tf.float32, shape=input_shape)
    tf_dm_input1 = tf.placeholder(tf.float32, shape=input_shape)
    tf_dm_input2 = tf.placeholder(tf.float32, shape=input_shape)
    tf_dm_input3 = tf.placeholder(tf.float32, shape=input_shape)
    tf_label_main = tf.placeholder(tf.int32, shape=label_shape)


    tf_dm_input_concat = tf.concat([tf_dm_input1, tf_dm_input2, tf_dm_input3], -1)
    
    tf_label_aux1 = add_maxpool3d('aux1_label', tf.expand_dims(tf.to_float(tf_label_main), -1), stride=4)
    tf_label_aux1 = tf.to_int32(tf.reshape(tf_label_aux1, label_shape1))
    tf_label_aux2 = add_maxpool3d('aux2_label', tf.expand_dims(tf.to_float(tf_label_main), -1), stride=2)
    tf_label_aux2 = tf.to_int32(tf.reshape(tf_label_aux2, label_shape2))
    
    in_filters = tf_t1_input.shape[-1]
    conv1_filters = 64
    t1_conv1a= add_conv3d('t1_conv1a', tf_t1_input, in_filters, conv1_filters, 
                                                train_phase=train_phase)

    t1_pool1 = add_maxpool3d('t1_pool1', t1_conv1a, stride=2)

    # conv2 
    conv2_filters = 128
    t1_conv2a = add_conv3d('t1_conv2a', t1_pool1, conv1_filters, conv2_filters,
                                                train_phase=train_phase )
    t1_pool2 = add_maxpool3d('t1_pool2', t1_conv2a, stride=2)

    # conv3(a+b) 
    conv3_filters = 256
    t1_conv3a = add_conv3d('t1_conv3a', t1_pool2, conv2_filters, conv3_filters,
                                                train_phase=train_phase)
    t1_conv3b = add_conv3d('t1_conv3b', t1_conv3a, conv3_filters, conv3_filters, 
                                                train_phase=train_phase)
    t1_pool3 = add_maxpool3d('t1_pool3', t1_conv3b, stride=2)

    #conv4(a+b)
    conv4_filters = 512
    t1_conv4a = add_conv3d('t1_conv4a', t1_pool3, conv3_filters, conv4_filters, 
                                                train_phase=train_phase)
    t1_conv4b = add_conv3d('t1_conv4b', t1_conv4a, conv4_filters, conv4_filters,
                                                train_phase=train_phase)
    # pool4 = add_maxpool3d('pool4', conv4b, stride=2)


    ###################### add t2-modality ############################

    in_filters = tf_t2_input.shape[-1]
    conv1_filters = 64
    t2_conv1a= add_conv3d('t2_conv1a', tf_t2_input, in_filters, conv1_filters, 
                                                train_phase=train_phase)

    t2_pool1 = add_maxpool3d('t2_pool1', t2_conv1a, stride=2)

    # conv2 
    conv2_filters = 128
    t2_conv2a = add_conv3d('t2_conv2a', t2_pool1, conv1_filters, conv2_filters,
                                                train_phase=train_phase )
    t2_pool2 = add_maxpool3d('t2_pool2', t2_conv2a, stride=2)

    # conv3(a+b) 
    conv3_filters = 256
    t2_conv3a = add_conv3d('t2_conv3a', t2_pool2, conv2_filters, conv3_filters,
                                                train_phase=train_phase)
    t2_conv3b = add_conv3d('t2_conv3b', t2_conv3a, conv3_filters, conv3_filters, 
                                                train_phase=train_phase)
    t2_pool3 = add_maxpool3d('t2_pool3', t2_conv3b, stride=2)

    #conv4(a+b)
    conv4_filters = 512
    t2_conv4a = add_conv3d('t2_conv4a', t2_pool3, conv3_filters, conv4_filters, 
                                                train_phase=train_phase)
    t2_conv4b = add_conv3d('t2_conv4b', t2_conv4a, conv4_filters, conv4_filters,
                                                train_phase=train_phase)


    ###################################################################


    ###################### add dm-modality ############################

    # in_filters = tf_dm_input.shape[-1]
    in_filters = tf_dm_input_concat.shape[-1]
    conv1_filters = 64
    dm_conv1a= add_conv3d('dm_conv1a', tf_dm_input_concat, in_filters, conv1_filters, 
                                            train_phase=train_phase)

    dm_pool1 = add_maxpool3d('dm_pool1', dm_conv1a, stride=2)

    # conv2 
    conv2_filters = 128
    dm_conv2a = add_conv3d('dm_conv2a', dm_pool1, conv1_filters, conv2_filters,
                                            train_phase=train_phase )
    dm_pool2 = add_maxpool3d('dm_pool2', dm_conv2a, stride=2)

    # conv3(a+b) 
    conv3_filters = 256
    dm_conv3a = add_conv3d('dm_conv3a', dm_pool2, conv2_filters, conv3_filters,
                                            train_phase=train_phase)
    dm_conv3b = add_conv3d('dm_conv3b', dm_conv3a, conv3_filters, conv3_filters, 
                                            train_phase=train_phase)
    dm_pool3 = add_maxpool3d('dm_pool3', dm_conv3b, stride=2)

    #conv4(a+b)
    conv4_filters = 512
    dm_conv4a = add_conv3d('dm_conv4a', dm_pool3, conv3_filters, conv4_filters, 
                                            train_phase=train_phase)
    dm_conv4b = add_conv3d('dm_conv4b', dm_conv4a, conv4_filters, conv4_filters,
                                            train_phase=train_phase)

    conv4b = tf.concat([t1_conv4b,t2_conv4b, dm_conv4b], -1)

    ###################################################################


    # deconv1(a+b+c)
    deconv1_filters = 256
    deconv1a = add_deconv3d('deconv1a', conv4b, conv4_filters*3, deconv1_filters, 
                                                stride=2 , train_phase=train_phase)
   
    concat1 = tf.concat([deconv1a, t1_conv3b], -1)
    
    deconv1b = add_conv3d('deconv1b', concat1, deconv1_filters*2, deconv1_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv1c = add_conv3d('deconv1c', deconv1b, deconv1_filters, deconv1_filters,
                                                filter_size=3,  train_phase=train_phase)

    # deconv2(a+b+c)
    deconv2_filters = 128
    deconv2a = add_deconv3d('deconv2a', deconv1c, deconv1_filters, deconv2_filters, 
                                                stride=2 , train_phase=train_phase)
    concat2 = tf.concat( [deconv2a, t1_conv2a], -1)
    deconv2b = add_conv3d('deconv2b', concat2, deconv2_filters*2, deconv2_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv2c = add_conv3d('deconv2c', deconv2b,deconv2_filters, deconv2_filters,
                                                filter_size=3,  train_phase=train_phase)



    # deconv3(a+b+c)
    deconv3_filters = 64
    deconv3a = add_deconv3d('deconv3a', deconv2c, deconv2_filters, deconv3_filters, #32?
                                                stride=2 , train_phase=train_phase)
    concat3 = tf.concat( [deconv3a, t1_conv1a], -1)
    deconv3b = add_conv3d('deconv3b', concat3, deconv3_filters*2, deconv3_filters,
                                                filter_size=1,  train_phase=train_phase)
    deconv3c = add_conv3d('deconv3c', deconv3b, deconv3_filters, deconv3_filters,
                                                filter_size=3,  train_phase=train_phase)

    #output
    # last layer before output, should not contain relu layer
    final_out_filters = FLAGS.cls_out

    #aux1 output 
    aux1_conv = add_conv3d('aux1_conv', deconv1c, deconv1_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)
    aux1_pred = add_conv3d('aux1_pred', aux1_conv, final_out_filters, final_out_filters,
                                                stride=1, train_phase=train_phase,
                                                with_batch_relu=False)

    

    #aux2 output 
    aux2_conv = add_conv3d('aux2_conv', deconv2c, deconv2_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)
    aux2_pred = add_conv3d('aux2_pred', aux2_conv, final_out_filters, final_out_filters,
                                                stride=1, train_phase=train_phase,
                                                with_batch_relu=False)
    

    #main output
    main_pred = add_conv3d('main_pred', deconv3c, deconv3_filters, final_out_filters,
                                                stride=1,  train_phase=train_phase,
                                                with_batch_relu=False)

    main_possibility = tf.nn.softmax(main_pred)



    new_vars = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))




    aux1_loss = cal_loss(aux1_pred, tf_label_aux1)
    aux2_loss = cal_loss(aux2_pred, tf_label_aux2)
    main_loss = cal_loss(main_pred, tf_label_main)

    final_loss = aux1_loss*FLAGS.aux1_weight + aux2_loss*FLAGS.aux2_weight \
                                             + main_loss*FLAGS.main_weight

    # l2 loss
    # for each varibale name, like:  conv2a/biases:0  and aux2_pred/weights:0
    for _var in gene_vars:
       
        # to use L2 loss, all the variables must be with the name of "weights"
        if _var.name[-9:-2] == 'weights': # to exclude 'bias' term
            final_loss = final_loss + FLAGS.L2_loss_weight*tf.nn.l2_loss(_var)
        
    
    return (tf_t1_input, tf_t2_input, tf_dm_input1, tf_dm_input2, tf_dm_input3, tf_label_main, 
            aux1_pred, aux2_pred, main_pred,
            aux1_loss, aux2_loss, main_loss, 
            final_loss, gene_vars, main_possibility)

'''
def create_optimizers(train_loss):
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    momentum = FLAGS.momentum
    train_opti = tf.train.MomentumOptimizer(learning_rate, momentum)
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    
    tvs = tf.trainable_variables()
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    
    gvs = train_opti.compute_gradients(train_loss, tvs)
    
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    
    train_minimize = train_opti.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)],
                                                name='apply_gradients',global_step=global_step)
    
    #train_minimize = train_opti.minimize(train_loss, name='loss_minimize', global_step=global_step)#, var_list=var_list)

    return zero_ops,accum_ops,train_minimize, learning_rate, global_step
'''