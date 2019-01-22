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
# train_config['train_json_file'] = '/media/ulsee/E/ai_format_dataset/total_ai_format_annos.json'
# train_config['train_data_path'] = '/media/ulsee/E/ai_format_dataset/imgs'
train_config['valid_batch_size']=1
train_config['train_nums']      = 372401
train_config['valid_nums']      = 30000
train_config['train_json_file'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
train_config['train_data_path'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'

# train params
train_config['batch_size'] = 36
train_config['height']     = 368
train_config['width']      = 368
train_config['num_kps']    = 14
train_config['paf']        = 26
train_config['sigma']      = 2.

train_config['paf_width_thre']  = 1.5
train_config['save_models']     = 100
train_config['input_scale']     = 4

# tri_hourglass
# train_config['log_path'] = 'logs/tri_hourglass.log'
# train_config['finetuning'] = None
# train_config['checkpoint_path'] = '/home/hsw/tri_hourglass/'

# hourglass
# train_config['log_path'] = 'logs/hourglass.log'
# train_config['finetuning'] = None
# train_config['checkpoint_path'] = '/home/hsw/hourglass/'

# head_neck_count_ori
train_config['log_path'] = 'logs/head_neck_count_2.log'
train_config['finetuning'] = None
train_config['checkpoint_path'] = '/home/hsw/head_neck_count_2/'