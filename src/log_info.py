# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: log_info.py
@time: 18-12-10 12:35
'''
import logging  # 引入logging模块
import os.path
import time

def get_logger():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    log_path = os.getcwd() + '/Logs/'
    log_name = log_path + 'openpose.log'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # print(log_path)

    logfile   = log_name
    fh        = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    # logger.warning('this ')
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y %m %d %H:%M', time.localtime(time.time()))
    logger.info('################################################################################')
    logger.info(rq)
    return logger

