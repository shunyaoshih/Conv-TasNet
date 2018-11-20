import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import librosa
import numpy as np
import logging

from dataloader import TasNetDataLoader
from tasnet import TasNet
from utils import *

if __name__ == '__main__':
    args, logger = setup()
    global_step = tf.Variable(0, trainable=False, name="global_step")
    if args.mode == 'train':
        train_dataloader = TasNetDataLoader("train", args.data_dir,
                                            args.batch_size, args.sample_rate)
        valid_dataloader = TasNetDataLoader("valid", args.data_dir,
                                            args.batch_size, args.sample_rate)
    else:
        infer_dataloader = TasNetDataLoader("infer", args.data_dir,
                                            args.batch_size, args.sample_rate)

    with tf.variable_scope("model") as scope:
        layers = {
            "conv1d_encoder":
            tf.keras.layers.Conv1D(
                filters=args.N,
                kernel_size=args.L,
                strides=args.L // 2,
                activation=tf.nn.relu,
                name="encode_conv1d"),
            "bottleneck":
            tf.keras.layers.Conv1D(args.B, 1, 1),
            "1d_deconv":
            tf.keras.layers.Dense(args.L, use_bias=False)
        }
        for i in range(2):
            layers["1x1_conv_decoder_{}".format(i)] = \
                tf.keras.layers.Conv1D(args.N, 1, 1)
        for r in range(args.R):
            for x in range(args.X):
                now_block = "block_{}_{}_".format(r, x)
                layers[now_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=args.H, kernel_size=1)
                layers[now_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=args.B, kernel_size=1)

        if args.mode == 'train':
            train_model = TasNet("train", train_dataloader, layers, 2, args.N,
                                 args.L, args.B, args.H, args.P, args.X,
                                 args.R)
            scope.reuse_variables()
            valid_model = TasNet("valid", valid_dataloader, layers, 2, args.N,
                                 args.L, args.B, args.H, args.P, args.X,
                                 args.R)
        else:
            infer_model = TasNet("infer", infer_dataloader, layers, 2, args.N,
                                 args.L, args.B, args.H, args.P, args.X,
                                 args.R)

    print_num_of_trainable_parameters()
    trainable_variables = tf.trainable_variables()

    valid_sdr = read_log(args.log_file)

    if args.mode == 'train':
        learning_rate = tf.placeholder(tf.float32, [])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = tf.gradients(train_model.loss, trainable_variables)
        update = opt.apply_gradients(
            zip(gradients, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state(args.log_dir)
        if ckpt:
            logging.info('Loading model from %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())

        if args.mode == 'train':
            lr = args.learning_rate
            valid_scores = [-1] * 2

            for epoch in range(1, args.max_epoch + 1):

                sess.run(train_dataloader.iterator.initializer)
                logging.info('-' * 20 + ' epoch {} '.format(epoch) + '-' * 25)

                train_iter_cnt, train_loss_sum = 0, 0
                while True:
                    try:
                        cur_loss, _, cur_global_step =\
                            sess.run(
                                fetches=[train_model.loss, update, global_step],
                                feed_dict={learning_rate: lr}
                            )
                        train_loss_sum += cur_loss * args.batch_size
                        train_iter_cnt += args.batch_size
                    except tf.errors.OutOfRangeError:
                        logging.info(
                            'step = {} , train SDR = {:5f} , lr = {:5f}'.
                            format(cur_global_step,
                                   -train_loss_sum / train_iter_cnt, lr))
                        break

                sess.run(valid_dataloader.iterator.initializer)
                valid_iter_cnt, valid_loss_sum = 0, 0
                while True:
                    try:
                        cur_loss, = sess.run([valid_model.loss])
                        valid_loss_sum += cur_loss * args.batch_size
                        valid_iter_cnt += args.batch_size
                    except tf.errors.OutOfRangeError:
                        cur_sdr = -(valid_loss_sum / valid_iter_cnt)

                        valid_scores.append(cur_sdr)
                        if max(valid_scores[-3:]) < valid_sdr:
                            lr /= 2

                        logging.info('validation SDR = {:5f}'.format(cur_sdr))
                        if cur_sdr > valid_sdr:
                            valid_sdr = cur_sdr
                            saver.save(
                                sess,
                                args.checkpoint_path,
                                global_step=cur_global_step)
                        break
        else:
            sess.run(infer_dataloader.iterator.initializer)
            infer_iter_cnt, infer_loss_sum = 0, 0
            while True:
                try:
                    cur_loss, outputs, single_audios, cur_global_step = sess.run(
                        fetches=[
                            infer_model.loss, infer_model.outputs, infer_model.
                            single_audios, global_step
                        ])

                    now_dir = args.log_dir + "/test/" + str(
                        infer_iter_cnt) + "/"

                    create_dir(now_dir)

                    outputs = [np.squeeze(output) for output in outputs]
                    single_audios = [
                        np.squeeze(single_audio)
                        for single_audio in single_audios
                    ]

                    def write(inputs, filename):
                        librosa.output.write_wav(
                            now_dir + filename,
                            inputs,
                            args.sample_rate,
                            norm=True)

                    # write(outputs[0], 's1.wav')
                    # write(outputs[1], 's2.wav')
                    # write(single_audios[0], 'true_s1.wav')
                    # write(single_audios[1], 'true_s2.wav')

                    infer_loss_sum += cur_loss * args.batch_size
                    infer_iter_cnt += args.batch_size

                    if infer_iter_cnt % 100 == 0:
                        print(-infer_loss_sum / infer_iter_cnt, infer_iter_cnt)
                except tf.errors.OutOfRangeError:
                    logging.info('step = {} , infer SDR = {:5f}'.format(
                        cur_global_step, -infer_loss_sum / infer_iter_cnt))
                    break
