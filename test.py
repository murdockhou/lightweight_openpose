# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: test.py
@time: 18-11-1 10:52
'''

import tensorflow as tf
import os
import cv2
import time
from src.parameters import params

from src.pose_decode import decode_pose
# from src.light_pose import create_light_weight_openpose
from src.light_openpose import create_light_weight_openpose


if __name__ == '__main__':

    use_gpu = True

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    cpm1, paf1, cpm2, paf2 = create_light_weight_openpose(input_img, heatmap_channels=params['num_keypoints'],
                                                          paf_channels=params['paf_channels'], is_training=False)

    saver = tf.train.Saver()
    param = {'thre1': 0.2, 'thre2': 0.05, 'thre3': 0.5}
    total_img = 0
    total_time = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params['test_model'])
        print('#---------Successfully loaded trained model.---------#')
        if params['video_path'] is not None:
            video_capture = cv2.VideoCapture(params['video_path'])
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            start_second = 0
            start_frame = fps * start_second
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while True:
                retval, img_data = video_capture.read()
                orih, oriw, c = img_data.shape
                # img_data = cv2.resize(img_data, (oriw // 2, oriw // 2))

                if not retval:
                    break
                img                                       = cv2.resize(img_data, (368,368)) / 255.
                # img = img_data / 255
                start_time                                = time.time()
                _1, _2, heatmap, paf                      = sess.run([cpm1, paf1, cpm2, paf2], feed_dict={input_img: [img]})
                end_time                                  = time.time()
                canvas, joint_list, person_to_joint_assoc = decode_pose(img, param, heatmap[0], paf[0])
                decode_time                               = time.time()
                # print('Inference time == {}, Decode time == {}'.format(end_time - start_time, decode_time - end_time))
                # canvas                                    = cv2.resize(canvas, (368*2,368*2))
                print ('inference + decode time == {}'.format(decode_time - start_time))
                total_img += 1
                total_time += (end_time-start_time)
                cv2.imshow('result', canvas)
                cv2.waitKey(0)
                # if total_img:
                #     print('Aver inference time === {}'.format(total_time / total_img))

        elif params['img_path'] is not None:
            for img_name in os.listdir(params['img_path']):
                img_data                                  = cv2.imread(os.path.join(params['img_path'], img_name))
                h, w, c = img_data.shape

                img                                       = cv2.resize(img_data, (params['width'], params['height'])) / 255.
                start_time                                = time.time()
                _1, _2, heatmap, paf                      = sess.run([cpm1, paf1, cpm2, paf2], feed_dict={input_img: [img]})
                end_time                                  = time.time()
                canvas, joint_list, person_to_joint_assoc = decode_pose(img, param, heatmap[0], paf[0])
                decode_time                               = time.time()
                # print('Inference time == {}, Decode time == {}'.format(end_time - start_time, decode_time - end_time))
                print('inference + decode time == {}'.format(decode_time - start_time))
                canvas                                    = cv2.resize(canvas, (img_data.shape[1], img_data.shape[0]))

                cv2.imshow('result', canvas)
                cv2.waitKey(0)

        else:
            print('Nothing to process.')


