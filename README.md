A tensorflow implementation about Arxiv Paper "[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/abs/1811.12004)"

**Thanks for the author provide PyTorch version**

(This repo is more useful than mine i think, enjoy it!)
Pytorch [implementation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch).

**trained model**

* upload 2019-02-13, in `model` folder, `model.ckpt-1008540.*`. 
* someone who wants to use this lightweight_openpose model on your own dataset, please train it by yourself. The trained model upload only use to test but not
good enough to use in practice. I did not trained it good enough. Pleas make sure that.

**update**
The original caffe prototxt(provided by paper author) has been upload, you can found in repo file named "lightweight_openpose.prototxt"

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

