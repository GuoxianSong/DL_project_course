#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import resnet


tf.app.flags.DEFINE_boolean('is_Server', True, """If is for server""")
tf.app.flags.DEFINE_boolean('is_Train', True, """If is for training""")
tf.app.flags.DEFINE_boolean('is_Simple',False,"""Test on simple model""")
# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """Number of GPUs.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.00005, """Initial learning rate""")

tf.app.flags.DEFINE_float('lr_decay', 0.95, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")


tf.app.flags.DEFINE_integer('dim_output',179,"""output dimension""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 150000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 10, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.8, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', 'train', """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS

os.chdir('/media/guoxian/D/3DFaceNetData/Coarse')


def get_lr(initial_lr, lr_decay, one_epoch_step, global_step):
    times_= int(global_step/one_epoch_step)
    return initial_lr*pow(lr_decay,times_)




def train():
    def Load():

        if(FLAGS.is_Simple or FLAGS.is_Train==False):
            train_images = np.load("Input/test_data.npy")
            train_labels = np.load("Input/test_label.npy")[:,:179]
        else:
            train_data = np.load("Input/train_data.npy")
            train_label = np.load("Input/train_label.npy")[:,:179]
            permutation = np.random.permutation(train_data.shape[0])
            train_images = train_data[permutation, :, :, :]
            train_labels = train_label[permutation, :]
        test_data  = np.load("Input/test_data.npy")
        test_label = np.load("Input/test_label.npy")[:,:179]
        mean_data  = np.array(np.load("Input/mean_data.npy"),dtype = np.float16)
        mean_label = np.load("Input/mean_label.npy")[:179]
        std_label   = np.load("Input/std_label.npy")[:179]

        return train_images,train_labels,test_data,test_label,mean_data,mean_label,std_label


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        X = tf.placeholder(tf.float32, [None, 224,224,3], name='Input_Images')
        Y = tf.placeholder(tf.float32, [None, FLAGS.dim_output], name='Input_labels')

        # Build model
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_output=FLAGS.dim_output,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)

        network_train = resnet.ResNet(hp, X, Y, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        print("sess 0")
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        print("sess 1")
        sess.run(init)
        print("sess done")
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        if (FLAGS.is_Train == False):

            checkpoint_dir = FLAGS.train_dir  # os.path.join(export_dir, 'checkpoint')
            checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoints and checkpoints.model_checkpoint_path:
                checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
                saver.restore(sess, os.path.join(checkpoint_dir, checkpoints_name))
            print('Load checkpoint %s' % checkpoints_name)

            init_step = global_step.eval(session=sess)
        else:
            print('Start from the scratch.')


        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

        # Training!
        train_images, train_labels, test_data, test_label, mean_data, mean_label,std_label= Load()
        one_epoch_step  =int(len(train_labels)/FLAGS.batch_size)
        train_images = (train_images-mean_data)
        train_labels = (train_labels-mean_label)/100.0

        test_data = (test_data-mean_data)
        test_label= (test_label - mean_label)/100.0
        print("data done")


        if(FLAGS.is_Train==False):

            max_iteration = int(len(test_label)/FLAGS.batch_size )
            print("max iteration is " +str(max_iteration))
            loss_=0
            for i in range(max_iteration):
                print(i)
                offset = (i * FLAGS.batch_size) % (test_data.shape[0] - FLAGS.batch_size)
                batch_data = test_data[offset:(offset + FLAGS.batch_size), :, :, :]
                batch_labels = test_label[offset:(offset + FLAGS.batch_size), :]
                logit_,loss_value = sess.run([ network_train.logits,network_train.loss],feed_dict={network_train.is_train: False,  X: batch_data,
                                        Y: batch_labels})
                loss_+=loss_value[0]
            print("test loss = " +str(loss_/ max_iteration))

        else:

            for step in range(init_step, FLAGS.max_steps):

                offset = (step * FLAGS.batch_size) % (train_labels.shape[0] - FLAGS.batch_size)
                batch_data = train_images[offset:(offset + FLAGS.batch_size), :, :,:]
                batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
                # Train
                lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, one_epoch_step, step)
                start_time = time.time()
                _, loss_value, train_summary_str = \
                        sess.run([network_train.train_op, network_train.loss,  train_summary_op],
                                feed_dict={network_train.is_train:True, network_train.lr:lr_value,X:batch_data,Y:batch_labels})
                duration = time.time() - start_time

                assert not np.isnan(loss_value)

                # Display & Summary(training)
                if step % FLAGS.display == 0 or step < 10:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: (Training) step %d, loss=%.4f, lr=%f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value, lr_value,
                                         examples_per_sec, sec_per_batch))

                    elapse = time.time() - start_time
                    time_left = (FLAGS.max_steps - step) * elapse
                    print("\tTime left: %02d:%02d:%02d" %
                          (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                    summary_writer.add_summary(train_summary_str, step)

                # Save the model checkpoint periodically.
                if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
