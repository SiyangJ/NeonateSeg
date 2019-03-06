import tensorflow as tf
import numpy as np
from pretrain_transfer import get_pretrained_weights
from util.utils import parse_patch_size,parse_class_weights

from config import FLAGS

DTYPE = tf.float32



def _weight_variable( scope_name, name, shape, from_pretrain=False, stddev=0.01):
    # with tf.device('/gpu:3'):
    if from_pretrain:
        print '>>> scope_name=', scope_name, " name=",name
        weights = get_pretrained_weights(scope_name, name,shape)
        if weights is None:
            if FLAGS.xavier_init:
                return tf.get_variable(name, shape, DTYPE, initializer=tf.contrib.layers.xavier_initializer())
            else:
                return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=stddev))
        else:
            init = tf.constant(weights)
        
        return tf.get_variable(name, initializer=init)
    else:
        if FLAGS.xavier_init:
            return tf.get_variable(name, shape, DTYPE, initializer=tf.contrib.layers.xavier_initializer())
        else:
            return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=stddev))



def _bias_variable( scope_name, name, shape, from_pretrain=False, constant_value=0.01):
    if from_pretrain:
        
        bias = get_pretrained_weights(scope_name, name,shape)
        if bias is None:
            return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(constant_value))
        else:
            init = tf.constant(bias)
            return tf.get_variable(name, initializer=init)
    else:
        bias = tf.get_variable(name, shape, DTYPE, tf.constant_initializer(constant_value))
        return bias

def cal_loss(logits, labels, calculate_weights=FLAGS.calculate_class_weights, additional_weights=None):
    num_classes = FLAGS.cls_out
    
    if calculate_weights:
        ## Enabled weights
        ## Test TODO
        assert td is not None
        labels_value = labels.eval()
        l_uni,l_counts = np.unique(labels_value,return_counts=True)
        weights = np.reciprocal(l_counts.astype(np.float32))
        classes_weights = weights * np.asarray(weights.shape,dtype=np.float32) / weights.sum()   
        ##
    else:
        if FLAGS.class_weights_string is not None:
            classes_weights = tf.constant(parse_class_weights(FLAGS.class_weights_string))
        else:
            classes_weights = tf.constant([ 1.0, 1.0,1.0,1.0])
    
    logits = tf.reshape(logits, (-1, num_classes))
    epsilon = tf.constant(value=1e-10)
    logits = logits + epsilon
    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
    if additional_weights is not None:
        labels = labels * tf.reshape(additional_weights, (-1,1))
    softmax = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), classes_weights), reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return loss

def add_conv3d(scope_name, prev_layer, in_filters, out_filters, 
                                        filter_size=3, 
                                        stride=1, 
                                        train_phase=True,
                                        with_batch_relu=True ):

    from_pretrain = FLAGS.from_pretrain
    with tf.variable_scope(scope_name) as scope:
        weights_shape = [filter_size, filter_size, filter_size, in_filters, out_filters]
        kernel = _weight_variable( scope_name, 'weights', weights_shape,from_pretrain=from_pretrain)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, stride, stride, stride, 1], padding='SAME')
        biases = _bias_variable( scope_name, 'biases', [out_filters],from_pretrain=from_pretrain)
        conv_bias = tf.nn.bias_add(conv, biases)

        if with_batch_relu is False:
            return conv_bias
        else:
            batch = tf.contrib.layers.batch_norm(conv_bias, center=True, scale=True,
                                                        # is_training=train_phase,
                                                        scope = scope_name)
            conv_output = tf.nn.relu(batch, name=scope.name)
            return conv_output


def add_deconv3d(scope_name, prev_layer, in_filters, out_filters, 
                                            filter_size=4, 
                                            stride=2,
                                            train_phase=True, 
                                            with_batch_relu=True):
    
    
    from_pretrain = FLAGS.from_pretrain
    with tf.variable_scope(scope_name) as scope:

        # for deconvolution, kernel[ 3*3*3* ouput * input_num]
        weights_shape = [filter_size, filter_size, filter_size, out_filters, in_filters]
        kernel = _weight_variable( scope_name, 'weights', weights_shape,from_pretrain=from_pretrain)

        num_example = FLAGS.batch_size if train_phase else 1
        output_shape = [num_example,
                        int(prev_layer.get_shape()[1]) * stride,
                        int(prev_layer.get_shape()[2]) * stride,
                        int(prev_layer.get_shape()[3]) * stride,
                        out_filters]

        conv = tf.nn.conv3d_transpose(prev_layer, kernel,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, stride, 1],
                                        padding='SAME')

        biases = _bias_variable( scope_name, 'biases', [out_filters], from_pretrain=from_pretrain)
        bias = tf.nn.bias_add(conv, biases)
        if with_batch_relu is False:
            return bias
        else:
            batch = tf.contrib.layers.batch_norm(bias, center=True, scale=True)
                                                        # is_training=train_phase )
            deconv_output = tf.nn.relu(batch, name=scope.name)

            return deconv_output



def add_maxpool3d(scope_name, prev_layer, stride=2, padding='SAME'):
    with tf.variable_scope(scope_name) as scope:
        pool_output = tf.nn.max_pool3d(prev_layer, ksize=[1, stride, stride, stride, 1], 
                                        strides=[1, stride, stride, stride, 1], 
                                        padding='SAME')

        return pool_output

###############################################################################
###### Editted by Siyang Jing of UNC on Nov 12
###### Only train part of the variables
###### Train first two conv stages, last deconv stage, and batch norms
def _get_var_list(freeze_encoder=False,
                  freeze_encoder_num=-1,
                  freeze_first=False):
    
    if freeze_encoder:
        trainable_var_list = tf.trainable_variables()
        TRAIN_NAME_LIST = ['aux','main','pred']
        if freeze_encoder_num == -1:
            TRAIN_NAME_LIST += ['deconv',]
        else:
            TRAIN_NAME_LIST += ['deconv%d'% (_i,) for _i in xrange(freeze_encoder_num+1,4)]
        to_train_list = []
        for v in trainable_var_list:
            for n in TRAIN_NAME_LIST:
                if n in v.name:
                    to_train_list.append(v)
                    break
    elif freeze_first:
        trainable_var_list = tf.trainable_variables()
        TRAIN_NAME_LIST = ['deconv3','_conv1','_conv2','gamma','beta','main','aux']
        to_train_list = []
        for v in trainable_var_list:
            for n in TRAIN_NAME_LIST:
                if n in v.name:
                    to_train_list.append(v)
                    break
    else:
        to_train_list = tf.trainable_variables()
    return to_train_list

def create_optimizers(train_loss):
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    momentum = FLAGS.momentum
    train_opti = tf.train.MomentumOptimizer(learning_rate, momentum)
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    
    if not FLAGS.restore_from_last:
        tvs = tf.trainable_variables()
    elif FLAGS.freeze_layers:
        tvs = _get_var_list(freeze_encoder=True,freeze_encoder_num=FLAGS.freeze_layers_num)
    else:
        tvs = tf.trainable_variables()
    
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    
    gvs = train_opti.compute_gradients(train_loss, tvs)
    
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    
    train_minimize = train_opti.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)],
                                                name='apply_gradients',global_step=global_step)
    
    #train_minimize = train_opti.minimize(train_loss, name='loss_minimize', global_step=global_step)#, var_list=var_list)

    return zero_ops,accum_ops,train_minimize, learning_rate, global_step
