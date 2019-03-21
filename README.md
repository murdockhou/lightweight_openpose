**update 2019-03-20**

Add evaluation part. The evaluation based on [ai-challenger evaluation](https://github.com/AIChallenger/AI_Challenger_2017/tree/master/Evaluation/keypoint_eval) solution. 
More information please refer to this link.

Easy way for how to use it:
* First, you need run  `test&model/model_json.py` to generate a json file. The `param` in this file contains everything you need.
Remember that `param['img_path] && param['jason_file']` parameters is the **groundtruth test files** that you need to test.
* Then, run `test&model/model_eval.py`, the command line would be like this:
    
        python model_eval.py --submit predictions.json --ref groundtruth.json
  this will give you a score that about your model performance, between **`0~1`**, 0 is worst and 1 is best.
  Make sure that you need run **`python2`** instead of **`python3`** because some errors will occur in `eval.py` file.



**update 2019-03-18**

Based on official pytorch implementation, re-write tf-model, see `lightweight_openpose.py` for detailed information, corresponding net structure picture is named `lightweight.png`. New pre_trained
model will be upload soon.
 
A tensorflow implementation about Arxiv Paper "[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/abs/1811.12004)"

**Thanks for the author provide PyTorch version**

(This repo is more useful than mine i think, enjoy it!)
Pytorch [implementation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch).

**trained model**

* ~~upload 2019-02-13, in `model` folder, `model.ckpt-1008540.*`.~~ 
* upload 2019-03-18, in `model`folder, named `model.ckpt-61236`, on ai_test_A dataset, get score **0.0377**, so bad.
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

**~~Note~~**

* ~~after training one epoch done, one validation epoch will be executed. So we set the `dataset.repeat(1)`. And use `make_initializable_iterator()` instead of `make_one_shot_iterator().~~
* ~~we modified some implementation about this model which inspired by this [article](https://arxiv.org/abs/1901.01760). we not only send previous stage output information to next stage as its input, but also used this previous output information to add next stage output featuremap in order to get next final output, which can be found in article.~~

