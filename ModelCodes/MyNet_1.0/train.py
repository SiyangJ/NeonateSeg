import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
from generator import get_training_and_testing_generators
from copy import deepcopy
from config import FLAGS
import re
from util.utils import parse_string_to_numbers

from predict_multimodality_sitk import eval_test_images_in_sitk

def _save_checkpoint(train_data, batch, log=True):
    td = train_data
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(batch))
    
    save_path = saver.save(td.sess, model_path) #, global_step=batch)
    if log:
        print("Model saved in file: %s" % save_path)
        print ('Model saved in file: %s' % saver.last_checkpoints)
        
def _rename_checkpoint(old_suffix, new_suffix):

    old_model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(old_suffix))
    new_model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(new_suffix))
    
    os.rename(old_model_path,new_model_path)

def _remove_one_checkpoint(batch):
    for suf in ['.data-00000-of-00001','.index','.meta']:
        ckpt_name = os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(batch)+suf)
        os.remove(ckpt_name)
    
def _remove_checkpoint_before(batch):
    sep = FLAGS.validate_every_n if FLAGS.save_around_best else FLAGS.checkpoint_period
    for b in xrange(0,batch,sep):
        if b==0:
            continue
        if os.path.exists(os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(b)+'.data-00000-of-00001')):
            _remove_one_checkpoint(b)
            
def _remove_checkpoint_after(batch):
    sep = FLAGS.validate_every_n if FLAGS.save_around_best else FLAGS.checkpoint_period
    batch += sep
    for b in xrange(batch,FLAGS.max_batch+1,sep):
        if b==0:
            continue
        if os.path.exists(os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(b)+'.data-00000-of-00001')):
            _remove_one_checkpoint(b)
            
def _remove_previous_not_needed(batch):
    assert FLAGS.save_around_best, 'Can only be used for save_around_best'
    _remove_checkpoint_before(batch - FLAGS.validate_every_n * FLAGS.save_around_num)

def _initialize_variables(train_data):
    td = train_data
    if FLAGS.restore_from_last:
        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(FLAGS.last_trained_checkpoint)
        print('**Training**: restore last checkpoint from:%s' % model_path)
        saver.restore(td.sess, model_path)
    else:
        try:
            init_op = td.sess.graph.get_operation_by_name('init')
        except KeyError as e:
            init_op = tf.global_variables_initializer()
        print('**Training**: global variable initialization...')
        td.sess.run(init_op)
        
def _eval_write(train_data,batch):
    td = train_data
    stats_mean = eval_test_images_in_sitk(td)
    
    cls_labels = list(parse_string_to_numbers(FLAGS.cls_labels,to_type=int))
    assert len(cls_labels)==FLAGS.cls_out, "Number of classes don't match"
    
    Dice_scores = [tf.Summary.Value(tag="Dice_%d"%_c, simple_value=stats_mean[_i]) for _i,_c in enumerate(cls_labels[1:])]
    
    summary = tf.Summary(value=Dice_scores)
    
    td.test_sum_writer.add_summary(summary, batch)
    print '>>> Epoch %d Test: [%3.3f, %3.3f, %3.3f]' % (batch,stats_mean[1],stats_mean[2],stats_mean[3])
    return stats_mean

## Rewrite the checkpoint file such that it always points to snapshot_best
def _rewrite_checkpoint_to_best():
    with open(os.path.join(FLAGS.checkpoint_dir,'checkpoint'),'r+') as _cf:
        _cs = _cf.read()
        _cs = re.sub(r'snapshot_.*?"','snapshot_best"',_cs)
        _cf.seek(0)
        _cf.write(_cs)
        _cf.truncate()

def _write_best_batch_to_file(best_batch):
    best_batch_file = os.path.join(FLAGS.checkpoint_dir,'best_batch')
    with open(best_batch_file,'w') as f:
        f.write('Best batch is:\n%d\n'%best_batch)

def _before_return(train_data,best_batch,record_best_batch=True):
    td = train_data
    _rewrite_checkpoint_to_best()
    if record_best_batch:
        _write_best_batch_to_file(best_batch)
    if FLAGS.save_around_best:
        _remove_checkpoint_after(best_batch + FLAGS.validate_every_n * FLAGS.save_around_num)
    if FLAGS.test_after_training:
        saver = tf.train.Saver()
        model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_best')
        print '>> **Test evaluation** after training: restore model from iteration %d at %s' % \
            (best_batch,model_path)
        saver.restore(td.sess, model_path)
        _ret = _eval_write(td,best_batch) 
        print _ret
        return _ret

def train_model(train_data):
    td = train_data

    # if the sess is restored from last checkpoint, do not need to 
    _initialize_variables(td)

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0

    assert FLAGS.learning_rate_reduce_life % 10 == 0

    training_generator, testing_generator = get_training_and_testing_generators(overwrite_split=FLAGS.overwrite_split)

    best_batch = -1
    best_lr = 1
    best_loss = 100
    fail_time = 0
    
    while not done:
        batch += 1
        print '>>>EPOCH %d' % batch
        # Update learning rate
        if batch % FLAGS.learning_rate_reduce_life == 0:
            lrval *= FLAGS.learning_rate_percentage
        
        if batch % FLAGS.validate_every_n == 0 or ((batch==1 or batch==2) and FLAGS.restore_from_last):
            
            ## Test evaluation info
            if FLAGS.show_test_in_training and \
                (batch % FLAGS.test_every_n == 0 or \
                ((batch==1 or batch==2) and FLAGS.restore_from_last)):
                
                _eval_write(td,batch)
            
            ## Training info
            total_aux1_loss = 0
            total_aux2_loss = 0
            total_main_loss = 0
            td.sess.run(td.zero_ops)
            for i in xrange(FLAGS.accumulate_times):
                if FLAGS.stage_1:
                    train_input_all = training_generator.next()
                    train_input1, train_input2, train_label = train_input_all[:3]
                    feed_dict = {td.tf_t1_input : train_input1, 
                                 td.tf_t2_input : train_input2,  
                                 td.tf_label: train_label, 
                                 td.learning_rate : lrval}
                else:
                    train_input_all = training_generator.next()
                    train_input1, train_input2, dm_input1, dm_input2, dm_input3, train_label = train_input_all[:6]
                    feed_dict = {td.tf_t1_input : train_input1, 
                     td.tf_t2_input : train_input2, 
                     td.tf_dm_input1: dm_input1,
                     td.tf_dm_input2: dm_input2, 
                     td.tf_dm_input3: dm_input3,
                     td.tf_label: train_label, 
                     td.learning_rate : lrval}
                if FLAGS.use_error_map:
                    feed_dict[td.tf_weight_main] = train_input_all[-1]
                ops = [td.accum_ops, td.aux1_loss, td.aux2_loss, td.main_loss]
                [_, aux1_loss, aux2_loss, main_loss] = td.sess.run(ops, feed_dict=feed_dict)
                
                total_aux1_loss += aux1_loss
                total_aux2_loss += aux2_loss
                total_main_loss += main_loss
                
            td.sess.run(td.train_minimize,feed_dict=feed_dict)
            
            total_aux1_loss /= FLAGS.accumulate_times
            total_aux2_loss /= FLAGS.accumulate_times
            total_main_loss /= FLAGS.accumulate_times
            
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="aux1_loss", simple_value=total_aux1_loss),
                tf.Summary.Value(tag="aux2_loss", simple_value=total_aux2_loss),
                tf.Summary.Value(tag="main_loss", simple_value=total_main_loss),
            ])
            td.summary_writer.add_summary(summary, batch)
            
            print("[%25s], Epoch: [%4d], Training Main Loss: [%3.3f], Lr[%1.8f]" 
                  % (time.ctime(), batch, total_main_loss,lrval))
            
            ## Validation info
            
            total_aux1_loss = 0
            total_aux2_loss = 0
            total_main_loss = 0
            for i in xrange(FLAGS.val_accumulate_times):               
                if FLAGS.stage_1:
                    val_input_all = testing_generator.next()
                    val_input1, val_input2, val_label = val_input_all[:3]
                    feed_dict = {td.tf_t1_input : val_input1, 
                                 td.tf_t2_input : val_input2,  
                                 td.tf_label: val_label}
                else:
                    val_input_all = testing_generator.next()
                    val_input1, val_input2, val_dm_1, val_dm_2, val_dm_3, val_label = val_input_all[:6]
                    feed_dict = {td.tf_t1_input : val_input1, 
                             td.tf_t2_input : val_input2,  
                             td.tf_dm_input1: val_dm_1,
                             td.tf_dm_input2: val_dm_2, 
                             td.tf_dm_input3: val_dm_3,
                             td.tf_label    : val_label}
                if FLAGS.use_error_map:
                    feed_dict[td.tf_weight_main] = val_input_all[-1]
                    
                ops = [td.aux1_loss, td.aux2_loss, td.main_loss]
                [aux1_loss, aux2_loss, main_loss] = td.sess.run(ops, feed_dict=feed_dict)
                
                total_aux1_loss += aux1_loss
                total_aux2_loss += aux2_loss
                total_main_loss += main_loss
            
            total_aux1_loss /= FLAGS.val_accumulate_times
            total_aux2_loss /= FLAGS.val_accumulate_times
            total_main_loss /= FLAGS.val_accumulate_times
            
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="aux1_loss", simple_value=total_aux1_loss),
                tf.Summary.Value(tag="aux2_loss", simple_value=total_aux2_loss),
                tf.Summary.Value(tag="main_loss", simple_value=total_main_loss),
            ])
            td.val_sum_writer.add_summary(summary, batch)  
            
            print("[%25s], Epoch: [%4d], Validation Main Loss: [%3.3f]" 
                  % (time.ctime(), batch, total_main_loss))  
            
            '''
            ## Save Around Model
            if FLAGS.save_around_best and batch % FLAGS.validate_every_n==0:
                for save_num in xrange(1,FLAGS.save_around_num+1):
                    before_batch = FLAGS.validate_every_n * (FLAGS.save_around_num - save_num)
                    if batch > before_batch:
                        _move_save_around_to_previous(FLAGS.save_around_num - save_num)
                _save_checkpoint(td, 'temp-0', log=False)
            '''
            
            ## Early stopping check
            
            if best_batch==-1 or total_main_loss < best_loss:
                ## Save model
                _save_checkpoint(td, 'best', log=False)
                best_batch = batch
                best_loss = total_main_loss
                best_lr = lrval
                fail_time = 0
                if FLAGS.save_around_best:
                    _remove_previous_not_needed(best_batch)
            elif batch-best_batch >= FLAGS.early_stop_iteration:
                
                if fail_time >= FLAGS.early_stop_max_fail:
                    print(">>> TRAINING STOP: From batch %d lr has been decreased %d times and loss does not decrease." % 
                          (best_batch,FLAGS.early_stop_max_fail))
                    print '>>> Best batch: [%4d], best loss: [%5.5f], lr: [%1.8f]' % (best_batch,best_loss,best_lr)
                    
                    return _before_return(td,best_batch)
                
                print '>> EARLY STOP: after %d iterations, still not decreasing' % FLAGS.early_stop_iteration
                ## Restore model
                saver = tf.train.Saver()
                model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_best')
                print '>> Restore model from iteration %d at %s' % (best_batch,model_path)                
                saver.restore(td.sess, model_path)
                batch = best_batch
                best_lr *= FLAGS.learning_rate_percentage
                lrval = best_lr
                fail_time += 1
            
        else:
            
            td.sess.run(td.zero_ops)
            for i in xrange(FLAGS.accumulate_times):
                if FLAGS.stage_1:
                    train_input_all = training_generator.next()
                    train_input1, train_input2, train_label = train_input_all[:3]
                    feed_dict = {td.tf_t1_input : train_input1, 
                                 td.tf_t2_input : train_input2,  
                                 td.tf_label: train_label, 
                                 td.learning_rate : lrval}
                else:
                    train_input_all = training_generator.next()
                    train_input1, train_input2, dm_input1, dm_input2, dm_input3, train_label = train_input_all[:6]
                    feed_dict = {td.tf_t1_input : train_input1, 
                     td.tf_t2_input : train_input2, 
                     td.tf_dm_input1: dm_input1,
                     td.tf_dm_input2: dm_input2, 
                     td.tf_dm_input3: dm_input3,
                     td.tf_label: train_label, 
                     td.learning_rate : lrval}
                if FLAGS.use_error_map:
                    feed_dict[td.tf_weight_main] = train_input_all[-1]
                ops = [td.accum_ops, td.main_loss]
                [_, main_loss] = td.sess.run(ops, feed_dict=feed_dict)
                
            td.sess.run(td.train_minimize,feed_dict=feed_dict)
            
        if FLAGS.save_around_best and batch % FLAGS.validate_every_n==0:
            _save_checkpoint(td, batch)
        elif batch % FLAGS.checkpoint_period == 0:
            _save_checkpoint(td, batch)
           
        if batch  >= FLAGS.max_batch:
            done = True

    _save_checkpoint(td, batch)
    
    return _before_return(td,best_batch)
    print('>>> Finished training!')
    print '>>> Best batch: [%4d], best loss: [%5.5f], lr: [%1.8f]' % (best_batch,best_loss,best_lr)


if __name__ == '__main__':
	import time
	time.ctime() # 'Mon Oct 18 13:35:29 2010'
	time.strftime('%l:%M%p %Z on %b %d, %Y') # ' 1:36PM EDT on Oct 18, 2010'
	time.strftime('%l:%M%p %z on %b %d, %Y') # ' 1:36PM EST on Oct 18, 2010'