# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: test_head_count.py
@time: 19-1-4 09:44
'''

import tensorflow as tf
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.lightweight_openpose import light_openpose
from src.pose_decode import decode_pose
# from src.pose_decode_old import decode_pose
params = {}
params['test_model'] = '/home/ulsee/server/lightweight_openpose/20190123-0355/model.ckpt-46552/model.ckpt-108619'
# params['video_path'] = '/media/ulsee/E/video/bank/jiachaojian30.mp4'
params['img_path']   = '/media/ulsee/E/ai_format_dataset/lsp_imgs'
# params['img_path']   = '/media/ulsee/E/yuncong/yuncong_data/our/test/0/'
params['score_threshold'] = 0.1
params['nms_threshold']   = 5

params['thre1'] = 0.1
params['thre2'] = 0.0


def coor_filter(heatmap_numpy, keep_coor, coordinate):
    radius = 2
    rows, cols = heatmap_numpy.shape
    ret_keep = []
    for index in keep_coor:
        x = int(coordinate[index][0])
        y = int(coordinate[index][1])
        left_col = max(0, y-radius)
        right_col = min(cols, y+radius+1)
        left_row = max(0, x-radius)
        right_row = min(rows, x+radius+1)
        clip_map = heatmap_numpy[left_row:right_row, left_col:right_col]
        clip_map = np.exp(clip_map) / np.sum(np.exp(clip_map))
        if np.max(clip_map) > 2*np.min(clip_map):
            ret_keep.append(index)

    return ret_keep


def pose_nms(img, heatmap, score_threshold, nms_threshold, type='multi'):

    img_height, img_width, _ = img.shape
    heatmap_height, heatmap_width, heatmap_channels = heatmap.shape
    factor_y = img_height / heatmap_height
    factor_x = img_width / heatmap_width
    joints_list = []

    if type == 'multi':
        heatmap_channels = 1
        for c in range(heatmap_channels):
            current_heatmap = heatmap[:, :, c]
            x, y = np.where(current_heatmap > score_threshold)
            coordinate = list(zip(x, y))
            scores = []
            for coor in coordinate:
                scores.append(current_heatmap[coor])
            s = np.asarray(scores)
            # print(s)
            s_index = s.argsort()[::-1]  # 降序，第一个位置的索引值最大
            # print(s_index)
            # nms
            keep = []

            while s_index.size > 0:
                keep.append(s_index[0])
                s_index = s_index[1:]
                last = []
                for index in s_index:
                    # print(keep[-1], index)
                    distance = np.sqrt(np.sum(np.square(
                        np.asarray(coordinate[keep[-1]]) - np.asarray(coordinate[index])
                    )))
                    if distance > nms_threshold:
                        last.append(index)

                s_index = np.asarray(last)

            # keep = coor_filter(current_heatmap, keep, coordinate)

            for index in keep:
                coor = coordinate[index]
                coorx = coor[0]
                coory = coor[1]

                coorx = int(coorx * factor_x)
                coory = int(coory * factor_y)

                cv2.circle(img, (coory, coorx), 5, (255, 0, 0), -1)
                cv2.putText(img, str(scores[index])[:4], (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                joints_list.append([coorx, coory])
    elif type == 'single':
        for c in range(heatmap_channels):
            current_heatmap = heatmap[:, :, c]

            cur_max = np.max(current_heatmap)
            if cur_max < score_threshold:
                continue
            index_all = np.where(current_heatmap == cur_max)
            coorx = index_all[0][0]
            coory = index_all[1][0]

            coorx = int(coorx * factor_x)
            coory = int(coory * factor_y)

            cv2.circle(img, (coory, coorx), 5, (0, 0, 255), -1)
            cv2.putText(img, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            joints_list.append([coorx, coory])
    else:
        print ('type must be multi or single.')

    return img, joints_list

def main():

    use_gpu = False

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    _1, _2, cpm, paf = light_openpose(input_img, is_training=False)

    saver = tf.train.Saver()

    total_img = 0
    total_time = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params['test_model'])
        print('#---------Successfully loaded trained model.---------#')
        if 'video_path'in params.keys() and params['video_path'] is not None:
            # video_capture = cv2.VideoCapture('rtsp://admin:youwillsee!@10.24.1.238')
            video_capture = cv2.VideoCapture(params['video_path'])
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            start_second = 0
            start_frame = fps * start_second
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while True:
                retval, img_data = video_capture.read()
                if not retval:
                    break
                img_data = cv2.cvtColor(img_data, code=cv2.COLOR_BGR2RGB)
                orih, oriw, c = img_data.shape
                # img = img_data / 255.
                # img = cv2.resize(img_data, (oriw // 3, oriw // 3)) / 255.
                img = cv2.resize(img_data, (368, 368)) / 255.
                start_time = time.time()
                heatmap, _paf = sess.run([cpm, paf], feed_dict={input_img: [img]})
                end_time = time.time()
                canvas, joints_list = pose_nms(img, heatmap[0], params['score_threshold'], params['nms_threshold'])
                # canvas, joint_list, person_to_joint_assoc = decode_pose(img, params, heatmap[0], _paf[0])
                decode_time = time.time()
                print ('inference + decode time == {}'.format(decode_time - start_time))
                total_img += 1
                total_time += (end_time-start_time)
                canvas = canvas.astype(np.float32)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                cv2.imshow('result', canvas)
                cv2.waitKey(1)
                # plt.imshow(canvas)
                # plt.axis('off')
                # plt.show()
        elif params['img_path'] is not None:
            for img_name in os.listdir(params['img_path']):
                if img_name.split('.')[-1] != 'jpg':
                    continue
                img_data = cv2.imread(os.path.join(params['img_path'], img_name))
                img_data = cv2.cvtColor(img_data, code=cv2.COLOR_BGR2RGB)
                orih, oriw, c = img_data.shape
                # img = img_data / 255.
                img = cv2.resize(img_data, (368, 368)) / 255.
                # img = cv2.resize(img_data, (320, 320)) / 255.
                start_time = time.time()
                heatmap, _paf = sess.run([cpm, paf], feed_dict={input_img: [img]})
                # pafx = _paf[0, ..., 0]
                # pafy = _paf[0, ..., 1]
                # pafx = (pafx - np.min(pafx)) / (np.max(pafx) - np.min(pafx))
                # pafy = (pafy - np.min(pafy)) / (np.max(pafy) - np.min(pafy))
                # cv2.imshow('paf0', pafx)
                # cv2.waitKey(0)
                # cv2.imshow('paf1', pafy)
                # cv2.waitKey(0)
                # print (heatmap.shape, _paf.shape)
                end_time = time.time()
                # canvas, joints_list = pose_nms(img, heatmap[0], params['score_threshold'], params['nms_threshold'])
                canvas, joint_list, person_to_joint_assoc = decode_pose(img, params, heatmap[0], _paf[0])
                # print (joint_list)
                # print (person_to_joint_assoc)
                canvas = canvas.astype(np.float32)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                decode_time = time.time()
                print('inference + decode time == {}'.format(decode_time - start_time))
                total_img += 1
                total_time += (end_time - start_time)
                cv2.imshow('result', canvas)
                cv2.waitKey(0)

        else:
            print('Nothing to process.')

main()