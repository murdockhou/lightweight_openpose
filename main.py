#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: main.py
@time: 2019/8/12 下午7:32
@desc:
'''
import tensorflow as tf
from train.kps_train import train
from nets.lightweight_openpose import lightweight_openpose
from data.dataset import get_dataset
from configs.ai_config import ai_config as params
import os
import datetime

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    inputs = tf.keras.Input(shape=(params['height'], params['width'], 3), name='model_input')
    outputs = lightweight_openpose(inputs, params['num_joints'], params['pafs'], True)
    model = tf.keras.Model(inputs, outputs)

    cur_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M')
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    dataset = get_dataset()
    epochs = 200
    summary_writer = tf.summary.create_file_writer(os.path.join('./logs/kps', cur_time))
    with summary_writer.as_default():
        train(model, optimizer, dataset, epochs, cur_time, max_keeps=epochs)