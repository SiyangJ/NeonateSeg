import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
from generator import get_training_and_testing_generators
from copy import deepcopy
from config import FLAGS

def _save_checkpoint(train_data, batch, log=True):
    td = train_data
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.checkpoint_dir,'snapshot_'+str(batch))
    
    save_path = saver.save(td.sess, model_path) #, global_step=batch)
    if log:
        print("Model saved in file: %s" % save_path)
        print ('Model saved in file: %s' % saver.last_checkpoints)


def train_model(train_data):
    td = train_data

    # if the sess is restored from last checkpoint, do not need to 
    if FLAGS.restore_from_last:
        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(FLAGS.last_trained_checkpoint)
        print('**Training**: restore last checkpoint from:%s' % model_path)
        saver.restore(td.sess, model_path)
    else:
        init_op = tf.global_variables_initializer()
        print('**Training**: global variable initialization...')
        td.sess.run(init_op)

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
        
        if batch % FLAGS.validate_every_n == 0:
            ## Training info
            total_aux1_loss = 0
            total_aux2_loss = 0
            total_main_loss = 0
            td.sess.run(td.zero_ops)
            for i in xrange(FLAGS.accumulate_times):
                if FLAGS.stage_1:
                    train_input1, train_input2, train_label = training_generator.next()
                    feed_dict = {td.tf_t1_input : train_input1, 
                                 td.tf_t2_input : train_input2,  
                                 td.tf_label: train_label, 
                                 td.learning_rate : lrval}
                else:
                    train_input1, train_input2, dm_input1, dm_input2, dm_input3, train_label = training_generator.next()
                    feed_dict = {td.tf_t1_input : train_input1, 
                     td.tf_t2_input : train_input2, 
                     td.tf_dm_input1: dm_input1,
                     td.tf_dm_input2: dm_input2, 
                     td.tf_dm_input3: dm_input3,
                     td.tf_label: train_label, 
                     td.learning_rate : lrval}
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
                    val_input1, val_input2, val_label = testing_generator.next()
                    feed_dict = {td.tf_t1_input : val_input1, 
                                 td.tf_t2_input : val_input2,  
                                 td.tf_label: val_label}
                else:
                    val_input1, val_input2, val_dm_1, val_dm_2, val_dm_3, val_label = testing_generator.next()
                    feed_dict = {td.tf_t1_input : val_input1, 
                             td.tf_t2_input : val_input2,  
                             td.tf_dm_input1: val_dm_1,
                             td.tf_dm_input2: val_dm_2, 
                             td.tf_dm_input3: val_dm_3,
                             td.tf_label    : val_label}
                
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
            
            
            ## Early stopping check
            
            if best_batch==-1 or total_main_loss < best_loss:
                ## Save model
                _save_checkpoint(td, 'best', log=False)
                best_batch = batch
                best_loss = total_main_loss
                best_lr = lrval
                fail_time = 0
            elif batch-best_batch >= FLAGS.early_stop_iteration:
                if fail_time >= FLAGS.early_stop_max_fail:
                    print(">>> TRAINING STOP: From batch %d lr has been decreased %d times and loss does not decrease." % 
                          (best_batch,FLAGS.early_stop_max_fail))
                    print '>>> Best batch: [%4d], best loss: [%5.5f], lr: [%1.8f]' % (best_batch,best_loss,best_lr)
                    return
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
                    train_input1, train_input2, train_label = training_generator.next()
                    feed_dict = {td.tf_t1_input : train_input1, 
                                 td.tf_t2_input : train_input2,  
                                 td.tf_label: train_label, 
                                 td.learning_rate : lrval}
                else:
                    train_input1, train_input2, dm_input1, dm_input2, dm_input3, train_label = training_generator.next()
                    feed_dict = {td.tf_t1_input : train_input1, 
                     td.tf_t2_input : train_input2, 
                     td.tf_dm_input1: dm_input1,
                     td.tf_dm_input2: dm_input2, 
                     td.tf_dm_input3: dm_input3,
                     td.tf_label: train_label, 
                     td.learning_rate : lrval}
                ops = [td.accum_ops, td.main_loss]
                [_, main_loss] = td.sess.run(ops, feed_dict=feed_dict)
                
            td.sess.run(td.train_minimize,feed_dict=feed_dict)
            
        if batch % FLAGS.checkpoint_period == 0:
            _save_checkpoint(td, batch)

        if batch  >= FLAGS.max_batch:
            done = True

    _save_checkpoint(td, batch)
    print('>>> Finished training!')
    print '>>> Best batch: [%4d], best loss: [%5.5f], lr: [%1.8f]' % (best_batch,best_loss,best_lr)


if __name__ == '__main__':
	import time
	time.ctime() # 'Mon Oct 18 13:35:29 2010'
	time.strftime('%l:%M%p %Z on %b %d, %Y') # ' 1:36PM EDT on Oct 18, 2010'
	time.strftime('%l:%M%p %z on %b %d, %Y') # ' 1:36PM EST on Oct 18, 2010'