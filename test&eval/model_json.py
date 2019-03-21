'''
create predict json file as ai-challenger format for model evaluation.
detail foramt adn information see:
    https://github.com/AIChallenger/AI_Challenger_2017
'''

import tensorflow as tf
import os
import time
import json
import cv2
import numpy as np

import sys
sys.path.append('../')


from src.lightweight_openpose import lightweight_openpose
from src.pose_decode import decode_pose

params = {}
params['test_model'] = '/home/hsw/work/github/lightweight_openpose/model/model.ckpt-61236'

params['img_path']   = '/media/hsw/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103'
params['json_file']  = '/media/hsw/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_annotations_20180103.json'
params['save_json']  = '/home/hsw/work/github/lightweight_openpose/test&eval/ai_test_a_predictions.json'

params['input_size'] = 256
params['score_threshold'] = 0.2
params['nms_threshold']   = 5
params['gpu']   = True
params['thre1'] = 0.1
params['thre2'] = 0.0


def main():

    use_gpu = params['gpu']

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(params['json_file'], encoding='utf-8') as fp:
        labels = json.load(fp)


    input_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    cpm, paf = lightweight_openpose(input_img, num_pafs=26, num_joints=14, is_training=False)

    saver = tf.train.Saver()
    total_img = 0

    predictions = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params['test_model'])
        print('#---------Successfully loaded trained model.---------#')

        for label in labels:

            img_id = label['image_id']

            img_ori  = cv2.imread(os.path.join(params['img_path'], img_id+'.jpg'))
            img_data = cv2.cvtColor(img_ori, code=cv2.COLOR_BGR2RGB)
            img_data = cv2.resize(img_data, (params['input_size'], params['input_size']))
            img = img_data / 255.

            heatmap, _paf  = sess.run([cpm, paf], feed_dict={input_img: [img]})
            canvas, joint_list, person_to_joint_assoc, joints = decode_pose(img_data, params, heatmap[0], _paf[0])

            predict = label
            predict['image_id'] = img_id
            kps = {}
            human = 1
            factorY = img_ori.shape[0] / img_data.shape[0]
            factorX = img_ori.shape[1] / img_data.shape[1]

            for joint in joints:
                for i in range(14):
                    joint[3*i]   *= factorX
                    joint[3*i+1] *= factorY
                if type(joint) == type([]):
                    kps['human' + str(human)] = joint
                else:
                    kps['human' + str(human)] = joint.tolist()
                human += 1

            # print (human)
            # print (kps)
            # for person, joint in kps.items():
            #     assert len(joint) == 14 * 3
            #     for i in range(14):
            #         x = int(joint[i*3])
            #         y = int(joint[i*3+1])
            #         cv2.circle(img_ori, (x, y), 3, (255, 255, 255), thickness=-1)
            #         cv2.putText(img_ori, str(i), (x, y),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            # cv2.imshow('test', img_ori)
            # cv2.waitKey(0)

            predict['keypoint_annotations'] = kps
            predictions.append(predict)

            # break
            total_img += 1
            print ('processing number: {}'.format(total_img))
            # break
    with open(params['save_json'], 'w') as fw:
        json.dump(predictions, fw)

if __name__ == '__main__':
    main()