# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: train_config.py
@time: 19-1-4 17:13
''' 

train_config = {}

# dataset parameters
train_config['train_json_file'] = '/home/hsw/hswData/label/ai_format_total_annos.json'
train_config['train_data_path'] = '/home/hsw/hswData/imgs'
train_config['valid_batch_size']=1
train_config['train_nums']      = 372401
train_config['valid_nums']      = 30000
train_config['valid_json_file'] = '/home/hsw/hswData/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
train_config['valid_data_path'] = '/home/hsw/hswData/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'

# train params
train_config['batch_size'] = 24
train_config['height']     = 368
train_config['width']      = 368
train_config['num_kps']    = 14
train_config['paf']        = 13*2
train_config['sigma']      = 1.

train_config['paf_width_thre']  = 1.5
train_config['save_models']     = 100
train_config['input_scale']     = 4


train_config['log_path'] = 'logs/lightweight_openpose.log'
train_config['finetuning'] = '/home/hsw/lightweight_openpose/20190123-0355/model.ckpt-46552'
train_config['checkpoint_path'] = '/home/hsw/lightweight_openpose/'
