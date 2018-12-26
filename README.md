A tensorflow implementation about Arxiv Paper "[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/abs/1811.12004)"

**Requirement**
* tensorflow >= 1.11.0
* python 3.5+
* cuda && cudnn

**Train**

``python3 train.py``

all parameters has been set in ``src/parameters.py``.

**Train Dataset**

we use **ai-challenger** format dataset, which can found in this [website](https://challenger.ai/competition/keypoint).

**Note**

* after training one epoch done, one validation epoch will be executed. So we set the `dataset.repeat(1)`. And use `make_initializable_iterator()` instead of `make_one_shot_iterator()`
* we only used to trian four points of one single person, `head, neck, left_shoulder, right_shoulder`. For 
people who wants to train all 14 points, just change the  `params['num_keypoints] and params['paf_channels]` in 
`parameters.py`. And also need to change detail code about `line 96 - 103` in `dataset.py` and `line 30 - 31 ` in `get_paf.py`.

