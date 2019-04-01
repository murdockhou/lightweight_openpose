# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: train.py
@time: 18-11-20 20:04
'''

from src.train_config import train_config

from src.lightweight_openpose import lightweight_openpose
from src.dataset import get_dataset_pipeline

import tensorflow as tf


def train_input_fn(parameters, epochs, mode='train'):

    dataset = get_dataset_pipeline(parameters, epochs, mode)
    return  dataset

def model_fn(features, labels, mode, params):

    # get model output
    features = tf.reshape(features, [-1, params['height'],params['width'], 3])
    gt_cpms = labels[..., :params['num_kps']]
    gt_pafs = labels[..., params['num_kps']:params['num_kps'] + params['paf']]
    mask = labels[..., params['num_kps'] + params['paf']:]
    mask = tf.reshape(mask, [-1, params['height']//params['scale'], params['width']//params['scale'], 1])

    cpm, paf = lightweight_openpose(inputs=features, num_joints=params['num_kps'], num_pafs=params['paf'], is_training=True)

    predictions = {
        'pred_heatmap': cpm,
        'pred_paf': paf
    }

    tf.summary.image('img', features, max_outputs=3)
    tf.summary.image('pred_hmap', tf.reduce_sum(cpm, axis=3, keepdims=True), max_outputs=3)
    tf.summary.image('gt_hmap', tf.reduce_sum(gt_cpms, axis=3, keepdims=True), max_outputs=3)
    tf.summary.image('gt_paf', tf.expand_dims(
        (gt_pafs[..., 0] - tf.reduce_min(gt_pafs[..., 0])) / (tf.reduce_max(gt_pafs[..., 0]) - tf.reduce_min(gt_pafs[..., 0])),
        axis=3
    ), max_outputs=3)
    tf.summary.image('pred_paf', tf.expand_dims(
        (paf[..., 0] - tf.reduce_min(paf[..., 0])) / (tf.reduce_max(paf[..., 0]) - tf.reduce_min(paf[..., 0])),
        axis=3
    ), max_outputs=3)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    cpm_mask = tf.concat([mask for i in range(params['num_kps'])], axis=-1)
    paf_mask = tf.concat([mask for i in range(params['paf'])], axis=-1)
    cpm = tf.where(cpm_mask > 0, cpm, cpm * 0)
    paf = tf.where(paf_mask > 0, paf, paf * 0)
    gt_cpms = tf.where(cpm_mask > 0, gt_cpms, gt_cpms * 0)
    gt_pafs = tf.where(paf_mask > 0, gt_pafs, gt_pafs * 0)
    loss = tf.nn.l2_loss(cpm - gt_cpms) + tf.nn.l2_loss(paf - gt_pafs) * 2

    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics_dict = {
            'heatmap': tf.metrics.mean_squared_error(labels=gt_cpms, predictions=predictions['pred_heatmap']),
            'paf': tf.metrics.mean_squared_error(labels=gt_pafs, predictions=predictions['pred_paf'])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics_dict
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # step lr
        # values = [params['lr'], 0.1*params['lr'], 0.01*params['lr'], 0.001*params['lr']]
        # boundaries = [params['train_nums']*50, params['train_nums']*100, params['train_nums']*150]
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        # constant lr
        learning_rate = tf.Variable(params['lr'], trainable=False, name='lr')

        tf.identity(learning_rate, name='lr')
        tf.summary.scalar('lr', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )



def main():
    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=20,
        intra_op_parallelism_threads=20,
        allow_soft_placement=True)

    session_config.gpu_options.allow_growth = True

    # distribution_strategy = tf.contrib.distribute.OneDeviceStrategy(device='/gpu:6')

    steps_per_epoch = train_config['train_nums'] // train_config['batch_size']

    run_config = tf.estimator.RunConfig(
        # train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_steps=steps_per_epoch,
        save_summary_steps=100,
        log_step_count_steps=100,
        keep_checkpoint_max=200
    )


    if train_config['finetuning'] is not None:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=train_config['finetuning'])
        model_dir = train_config['finetuning']
    else:
        ws = None
        model_dir = train_config['checkpoint_path']

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, model_dir=model_dir,
        params={
            'batch_size': train_config['batch_size'],
            'train_nums': steps_per_epoch,
            'lr':4e-5,
            'height': train_config['height'],
            'width': train_config['width'],
            'num_kps': train_config['num_kps'],
            'paf': train_config['paf'],
            'scale': train_config['input_scale']
        },
        warm_start_from=ws
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('start training')

    train_spec = tf.estimator.TrainSpec(input_fn=lambda : train_input_fn(parameters=train_config, epochs=200, mode='train'))
    eval_spec  = tf.estimator.EvalSpec(input_fn=lambda : train_input_fn(parameters=train_config, epochs=1, mode='valid'))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
