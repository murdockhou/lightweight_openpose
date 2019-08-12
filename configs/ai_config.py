#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: ai_config.py
@time: 2019/8/12 下午6:01
@desc:
'''

ai_config = {}

ai_config['height'] = 256
ai_config['width'] = 256
ai_config['scale'] = 8
ai_config['batch_size'] = 16

ai_config['num_joints'] = 14
ai_config['sigma'] = 1.5
ai_config['pafs'] = 13 * 2
ai_config['paf_width'] = 1.5

ai_config['train_img_path'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
ai_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
# ai_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/train10.json'

ai_config['finetune'] = None
ai_config['ckpt'] = '/media/hsw/E/ckpt/lightweight_openpose'