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
params['epoch']      = 10
params['height']     = 368
params['width']      = 368
params['num_keypoints']   = 4
params['paf_channels']    = 6
params['input_scale']     = 8
params['train_nums']      = 56 * 30
params['valid_nums']      = 30000
params['train_json_file'] = '/media/ulsee/E/ai_format_dataset/total_ai_format_annos.json'
params['train_data_path'] = '/media/ulsee/E/ai_format_dataset/imgs'
params['valid_batch_size'] = 1
params['valid_json_file'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
params['valid_data_path'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
# params['train_nums'] = 1902
# params['valid_nums'] = 1902
# params['train_json_file'] = '/media/ulsee/E/ai_format_dataset/ai_format_lsp_new.json'
# params['train_data_path'] = '/media/ulsee/E/ai_format_dataset/lsp_imgs'
# params['valid_json_file'] = '/media/ulsee/E/ai_format_dataset/ai_format_lsp_new.json'
# params['valid_data_path'] = '/media/ulsee/E/ai_format_dataset/lsp_imgs'
# params['train_json_file'] = '/media/ulsee/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
# params['train_data_path'] = '/media/ulsee/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
# dataset parameters
params['buffer_size']    = 1000
params['sigma']          = 1.
params['paf_width_thre'] = 1

# model
params['finetuning']      =  '/media/ulsee/E/half_body/lighter_pose/20181219-1059/model.ckpt-6650'
params['checkpoint_path'] = '/media/ulsee/E/half_body/lighter_pose'
params['checkpoint_path_keras'] = '/media/ulsee/D/half_body/keras'

# test params
params['used_gpu'] = True
params['video_path'] = '/media/ulsee/E/video/111.ts'
# params['video_path'] = '/home/ulsee/work/ulsActionRecognition/data/video_fall2.mp4'
# params['video_path'] = None
params['img_path'] = '/media/ulsee/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
# params['img_path'] = '/media/ulsee/E/datasets/coco2017/val2017'
# params['test_model'] = '/media/ulsee/D/half_body/20181101-1742/model.ckpt-41102'
params['test_model'] = '/media/ulsee/E/half_body/lighter_pose/20181214-1545/model_alter.ckpt-7757'
