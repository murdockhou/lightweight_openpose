# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: parameters.py
@time: 18-10-30 16:40
'''

params = dict()

# train params
params['lr']         = 1e-4
params['batch_size'] = 56
params['height']     = 368
params['width']      = 368
params['num_keypoints']   = 4
params['paf_channels']    = 6
params['input_scale']     = 8
params['train_nums']      = 372401
params['valid_nums']      = 30000
params['train_json_file'] = '/media/ulsee/E/ai_format_dataset/total_ai_format_annos.json'
params['train_data_path'] = '/media/ulsee/E/ai_format_dataset/imgs'
params['valid_batch_size'] = 4
params['valid_json_file'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
params['valid_data_path'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
params['log_path']        = 'Logs/lightweight_openpose.log'

# dataset parameters
params['sigma']          = 1.
params['paf_width_thre'] = 1

# model
params['finetuning']      =  None
params['checkpoint_path'] = '/media/ulsee/E/half_body/lightweight_openpose'

