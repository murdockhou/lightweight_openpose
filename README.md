A tensorflow implementation about Arxiv Paper "[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/abs/1811.12004)"

**Note**

The original caffe prototxt(provided by paper author) has been upload, you can found in repo file named `lightweight_openpose.prototxt`

**Requirement**
* tensorflow >= 1.11.0
* python 3.6+
* cuda && cudnn
* [imgaug](https://github.com/aleju/imgaug)

**Train**

``python3 train.py``

all parameters has been set in ``src/train_config.py``.

**Train Dataset**

we use **ai-challenger** format dataset, which can found in this [website](https://challenger.ai/competition/keypoint).

**Note**

* after training one epoch done, one validation epoch will be executed. So we set the `dataset.repeat(1)`. And use `make_initializable_iterator()` instead of `make_one_shot_iterator()`
* we modified some implementation about this model which inspired by this [article](https://arxiv.org/abs/1901.01760). we not only send previous stage output information to next stage as its input, but also used this previous output information to add next stage output featuremap in order to get next final output, which can be found in article.

