from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import tensorflow as tf

import numpy as np
from crnn_model import model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
_IMAGE_HEIGHT = 32
_IMAGE_WIDTH = 128

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 8, 'The number of threads to use in batch shuffling') 

tf.app.flags.DEFINE_integer(
    'step_per_eval', 100, 'The number of training steps to run between evaluations.')

tf.app.flags.DEFINE_integer(
    'step_per_test', 1000, 'The number of training steps to run between evaluations.')

tf.app.flags.DEFINE_integer(
    'step_per_save', 1000, 'The number of training steps to run between save checkpoints.')

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_train_steps', 20000, 'The number of maximum iteration steps for training')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.1, 'The initial learning rate for training.')

tf.app.flags.DEFINE_integer(
    'decay_steps', 1000, 'The learning rate decay steps for training.')

tf.app.flags.DEFINE_float(
    'decay_rate', 0.8, 'The learning rate decay rate for training.')

# ------------------------------------LSTM prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'lstm_hidden_layers', 2, 'The number of stacked LSTM cell.')

tf.app.flags.DEFINE_integer(
    'lstm_hidden_uints', 256, 'The number of units in each LSTM cell')

tf.app.flags.DEFINE_integer(
    'crop_height', 32, 'The height of crop size in image')
tf.app.flags.DEFINE_integer(
    'crop_width', 128, 'The width of crop size in image')

# ------------------------------------Char dictionary------------------------------------

tf.app.flags.DEFINE_string(
    'char_map_json_file', './char_map/plate_map.json', 'Path to char map json file')

FLAGS = tf.app.flags.FLAGS

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _read_train_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannot find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    max_delta = 50
    contrast_lower = 0.5
    contrast_upper = 1.5
    images = tf.image.decode_jpeg(features['images'])
    tf.image.random_contrast(images, contrast_lower, contrast_upper, seed=None)
    images = tf.image.random_brightness(images, max_delta, seed=None)
    images = tf.image.resize_with_crop_or_pad(images,FLAGS.crop_height,FLAGS.crop_width)
    images.set_shape([FLAGS.crop_height, FLAGS.crop_width, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 8, tf.int32)
    imagenames = features['imagenames']
    return images, labels, sequence_length, imagenames

def _read_test_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannot find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.image.decode_jpeg(features['images'])
    images = tf.image.resize_images(images,FLAGS.crop_height,FLAGS.crop_width)
    images.set_shape([_IMAGE_HEIGHT, _IMAGE_WIDTH, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 8, tf.int32)
    imagenames = features['imagenames']
    return images, labels, sequence_length, imagenames


def _train_crnn_ctc():
    train_tfrecord_path = os.path.join(FLAGS.data_dir, 'train.tfrecord')
    train_images, train_labels, train_sequence_lengths, _ = _read_train_tfrecord(tfrecord_path=train_tfrecord_path)
    test_tfrecord_path = os.path.join(FLAGS.data_dir, 'validation.tfrecord')
    test_images, test_labels, test_sequence_lengths, test_imagenames = _read_test_tfrecord(tfrecord_path=test_tfrecord_path)

    # get the test iter size
    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(test_tfrecord_path):
        test_sample_count += 1
    step_nums = test_sample_count // FLAGS.batch_size

    # decode the training data from tfrecords
    train_batch_images, train_batch_labels, train_batch_sequence_lengths = tf.train.batch(
        tensors=[train_images, train_labels, train_sequence_lengths], batch_size=FLAGS.batch_size, dynamic_pad=True,
        capacity=1000 + 2*FLAGS.batch_size, num_threads=FLAGS.num_threads)
    # decode the testing data from tfrecords
    test_batch_images, test_batch_labels, test_batch_sequence_lengths, test_batch_imagenames = tf.train.batch(
        tensors=[test_images, test_labels, test_sequence_lengths, test_imagenames], batch_size=FLAGS.batch_size, dynamic_pad=True,
        capacity=1000 + 2*FLAGS.batch_size, num_threads=FLAGS.num_threads)

    input_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='input_sequence_lengths')

    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='train',
                                    hidden_num=FLAGS.lstm_hidden_uints,
                                    layers_num=FLAGS.lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)
 
    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(images=input_images, sequence_length=input_sequence_lengths)

    ctc_loss = tf.reduce_mean(
        tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=input_sequence_lengths,
            ignore_longer_outputs_than_inputs=True))

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_lengths, merge_repeated=False)

    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step)

    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(FLAGS.model_dir, model_name)  

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(FLAGS.max_train_steps):
            if (step + 1) % FLAGS.step_per_test == 0 or step == 0:
                accuracy = []
                for _ in range(step_nums):
                    imgs, lbls, seq_lens, names = sess.run([test_batch_images, test_batch_labels, test_batch_sequence_lengths, test_batch_imagenames])
                    preds = sess.run(ctc_decoded, feed_dict={input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens})
                    preds = _sparse_matrix_to_list(preds[0], char_map_dict)
                    lbls = _sparse_matrix_to_list(lbls, char_map_dict)
                    #print(preds)
                    # #print(lbls)
                    for index, lbl in enumerate(lbls):
                        pred = preds[index]
                        total_count = len(lbl)
                        correct_count = 0
                        try:
                            for i, tmp in enumerate(lbl):
                                if tmp == pred[i]:
                                    correct_count += 1
                        except IndexError:
                            continue
                        finally:
                            try:
                                accuracy.append(correct_count / total_count)
                            except ZeroDivisionError:
                                if len(pred) == 0:
                                    accuracy.append(1)
                                else:
                                    accuracy.append(0)
                    for index, img in enumerate(imgs):
                        print('Predict {:s} image with gt label: {:s} <--> predict label: {:s}'.format(str(names[index]), str(lbls[index]), str(preds[index])), flush=True)
                        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                        print('Mean test accuracy is {:5f}'.format(accuracy), flush=True)


            imgs, lbls, seq_lens = sess.run([train_batch_images, train_batch_labels, train_batch_sequence_lengths])

            _, cl, lr, sd, preds, summary = sess.run(
                [optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded, merge_summary_op],
                feed_dict = {input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens})

            if (step + 1) % FLAGS.step_per_save == 0: 
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % FLAGS.step_per_eval == 0:
                # calculate the precision
                preds = _sparse_matrix_to_list(preds[0], char_map_dict)
                gt_labels = _sparse_matrix_to_list(lbls, char_map_dict)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    total_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / total_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                print('step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, sd, accuracy),  flush=True)

            
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

def main(unused_argv):
    _train_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 
