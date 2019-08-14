from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf

# set log level
tf.logging.set_verbosity(tf.logging.INFO)

from src.models import cnnmodel
from tools.config import config, generate_config, default

def parse_args():
    parser = argparse.ArgumentParser(description="Train face network")
    # general
    parser.add_argument("--network", default=default.network, help="network config")
    #parser.add_argument("--loss", default=default.loss, help="loss config")
    #parser.add_argument("--dataset", default=default.dataset, help="dataset")
    args, rest = parser.parse_known_args()
    generate_config(args.network)
    args = parser.parse_args()
    return args

parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=default.CUDA_VISIBLE_DEVICES
_IMAGE_HEIGHT = config.image_height
_IMAGE_WIDTH = config.image_width

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(default.char_map_json_file, 'r'))
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
        char_map_dict = json.load(open(default.char_map_json_file, 'r'))
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
    max_delta = 20
    contrast_lower = 0.8
    contrast_upper = 1.2
    images = tf.image.decode_jpeg(features['images'])
    tf.image.random_contrast(images, contrast_lower, contrast_upper, seed=None)
    images = tf.image.random_brightness(images, max_delta, seed=None)
    images = tf.image.resize_with_crop_or_pad(images,_IMAGE_HEIGHT,_IMAGE_WIDTH)
    images.set_shape([_IMAGE_HEIGHT, _IMAGE_WIDTH, 3])
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
    images = tf.image.resize_images(images,(_IMAGE_HEIGHT,_IMAGE_WIDTH))
    images.set_shape([_IMAGE_HEIGHT, _IMAGE_WIDTH, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 8, tf.int32)
    imagenames = features['imagenames']
    return images, labels, sequence_length, imagenames

def train_crnn_ctc():
    train_tfrecord_path = os.path.join(config.data_store_path, 'train.tfrecord')
    train_images, train_labels, train_sequence_lengths, _ = _read_train_tfrecord(tfrecord_path=train_tfrecord_path)
    trainval_targets = config.trainval_targets
    step_nums, accuracys, accuracys_max = [], dict(), dict()
    test_datasets_name, test_images, test_labels, test_sequence_lengths, test_imagenames = [], [], [] ,[], []
    for dataset_name in trainval_targets:
        test_tfrecord_path = os.path.join(config.data_store_path, '{}.tfrecord'.format(dataset_name))
        if os.path.exists(test_tfrecord_path):
            test_image, test_label, test_sequence_length, test_imagename = _read_test_tfrecord(tfrecord_path=test_tfrecord_path)
            test_datasets_name.append(dataset_name)
            test_images.append(test_image)
            test_labels.append(test_label)
            test_sequence_lengths.append(test_sequence_length)
            test_imagenames.append(test_imagename)
            accuracys_max[dataset_name] = -1.0
            # get the test iter size
            test_sample_count = 0
            for _ in tf.python_io.tf_record_iterator(test_tfrecord_path):
                test_sample_count += 1
            step_nums.append(test_sample_count // config.batch_size)




    # decode the training data from tfrecords
    train_batch_images, train_batch_labels, train_batch_sequence_lengths = tf.train.batch(
        tensors=[train_images, train_labels, train_sequence_lengths], batch_size=config.batch_size, dynamic_pad=True,
        capacity=1000 + 2*config.batch_size, num_threads=default.num_threads)
    # decode the testing data from tfrecords
    test_batch_images, test_batch_labels, test_batch_sequence_lengths, test_batch_imagenames = [], [], [], []
    for id, _ in enumerate(test_datasets_name):
        test_batch_image, test_batch_label, test_batch_sequence_length, test_batch_imagename = tf.train.batch(
            tensors=[test_images[id], test_labels[id], test_sequence_lengths[id], test_imagenames[id]], batch_size=config.batch_size, dynamic_pad=True,
            capacity=1000 + 2*config.batch_size, num_threads=default.num_threads)
        test_batch_images.append(test_batch_image)
        test_batch_labels.append(test_batch_label)
        test_batch_sequence_lengths.append(test_batch_sequence_length)
        test_batch_imagenames.append(test_batch_imagename)

    input_images = tf.placeholder(tf.float32, shape=[config.batch_size, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[config.batch_size], name='input_sequence_lengths')

    char_map_dict = json.load(open(default.char_map_json_file, 'r'))
    # initialise the net model

    with tf.variable_scope('CRNN_CTC', reuse=False):
        training = tf.placeholder(tf.bool, name='training')
        net_out = cnnmodel.build_network(input_images,  len(char_map_dict.keys()) + 1, training)
        #net_out = resmodel.build_network(input_images,  len(char_map_dict.keys()) + 1, training)
        #net_out = loccnnmodel.build_network(input_images, len(char_map_dict.keys()) + 1, training)

    ctc_loss = tf.reduce_mean(
        tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=input_sequence_lengths,
            ignore_longer_outputs_than_inputs=True))

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_lengths, merge_repeated=False)

    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(default.learning_rate, global_step, default.decay_steps, default.decay_rate, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        var_list = None
        if config.fixed_param is not None and config.fixed_param:
            var_list = []
            for sc in config.train_weights:
                sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
                var_list.extend(sc_variable)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step, var_list=var_list)

    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver()
    restore_path = tf.train.latest_checkpoint(default.model_save_path)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=config.train_weights)
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(restore_path, variables_to_restore)
    if not os.path.exists(default.save_root_path):
        os.makedirs(default.save_root_path)
    if not os.path.exists(default.model_save_path):
        os.makedirs(default.model_save_path)
    if not os.path.exists(default.model_stn_save_path):
        os.makedirs(default.model_stn_save_path)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    if config.stn:
        model_save_path = os.path.join(default.model_stn_save_path, model_name)
    else:
        model_save_path = os.path.join(default.model_save_path, model_name)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(model_save_path)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)
        if config.stn:
            init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_max_acc = -1.0
        tf.logging.info("+++++++++++++++++++ Start to train +++++++++++++++++++++++")
        tf.logging.info("Top of {} train steps".format(default.max_train_steps))
        tf.logging.info("{} step per test".format(default.step_per_test))
        tf.logging.info("{} step per eval".format(default.step_per_eval))
        tf.logging.info("{} step per save".format(default.step_per_save))
        tf.logging.info("----------------------------------------------------------")
        for step in range(default.max_train_steps):
            if (step + 1) % config.step_per_test == 0 or step == 0:
                for id, _ in test_datasets_name:
                    accuracy = []
                    for _ in range(step_nums[id]):
                        imgs, lbls, seq_lens, names = sess.run([test_batch_images[id], test_batch_labels[id], test_batch_sequence_lengths[id], test_batch_imagenames[id]])
                        preds = sess.run(ctc_decoded, feed_dict={input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens, training:False})
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
       #             for index, img in enumerate(imgs):
       #                 print('Predict {:s} image with gt label: {:s} <--> predict label: {:s}'.format(str(names[index]), str(lbls[index]), str(preds[index])), flush=True)
                    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                    if accuracy > accuracys_max[test_datasets_name[id]]:
                        accuracys_max[test_datasets_name[id]] = accuracy
                    tf.logging.info('Test dataset {} , Mean test accuracy is {:5f}, Max test accuracy is: {:5f}'.format(test_datasets_name[id], accuracy, accuracys_max[test_datasets_name[id]]))


            imgs, lbls, seq_lens = sess.run([train_batch_images, train_batch_labels, train_batch_sequence_lengths])

            _, cl, lr, sd, preds, summary = sess.run(
                [optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded, merge_summary_op],
                feed_dict = {input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens, training:True})

            if (step + 1) % default.step_per_save == 0:
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % default.step_per_eval == 0:
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

                tf.logging.info('step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, sd, accuracy))

            
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

def main():
    train_crnn_ctc()

if __name__ == '__main__':
    main()
