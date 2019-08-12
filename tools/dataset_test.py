#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: dataset_test.py
@time: 2019/8/12 下午6:06
@desc:
'''

from data.dataset import get_dataset

# dataset = get_dataset()
# for epoch in range(10):
#     for step, (img, heatmap, paf) in enumerate(dataset):
#         print ('step {}/{}'.format(step, epoch))
#         print (img.shape)
#         print (heatmap.shape)
#         print (paf.shape)
#         break
#     break

from utils.utils import *
from utils.data_aug import data_aug
import cv2, os
json_file = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
img_path = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
img_ids, id_kps_dict = read_json(json_file)

colors = [[0,0,255],[255,0,0],[0,255,0]]
for img_id in img_ids:
    print ('--------------------------------------------------------------')
    kps = id_kps_dict[img_id]
    # print('original, box:')

    img = cv2.imread(os.path.join(img_path, img_id + '.jpg'))

    img, kps = data_aug(img, None, kps)
    # continue
