A tensorflow implementation(2.0 version) about Arxiv Paper "[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/abs/1811.12004)"

**Thanks for the author provide PyTorch version**

(This repo is more useful than mine i think, enjoy it!)
Pytorch [implementation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch).

**Requirement**
* tensorflow >= 2.0
* python 3.6+
* cuda && cudnn
* [imgaug](https://github.com/aleju/imgaug)

**Train**

``python3 main.py``

all parameters has been set in ``configs/ai_config.py``.

**Train Dataset**

we use **ai-challenger** format dataset, which can found in this [website](https://challenger.ai/competition/keypoint).
