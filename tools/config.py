import numpy as np
import os
from easydict import EasyDict as edict

config = edict()
config.image_height = 64
config.image_width = 128

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.memonger = False #not work now


# network settings
network = edict()

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_output = 'E'
network.y2.net_blocks = [2,8,16,4]
#network.y2.net_blocks = [2,2,2,2]

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.image_list_path = "./data/anno_file.txt"
dataset.dataset_path = "./data/CCPD2019/ccpd_base"
dataset.data_save_path = "./data/images"

# default settings
default = edict()
default.CUDA_VISIBLE_DEVICES = "0"
default.char_map_json_file = "./char_map/plate_map.json"
default.dataset = "data_dir"
default.image_dir = './data/images/'
default.anno_file = './data/anno_file.txt'
default.data_dir = './data/'
default.validation_split_fraction = 0.1
default.shuffle_list = True

default.model_dir = './model/'
default.num_threads = 8
default.step_per_eval = 500
default.step_per_test = 3000
default.step_per_save = 3000
default.batch_size = 32
default.max_train_steps = 200000
default.learning_rate = 0.1
default.decay_steps = 10000
default.decay_rate = 0.8
default.lstm_hidden_layers = 2
default.lstm_hidden_uints = 256

# default network
default.network = 'y2'
default.pretrained = ''
default.pretrained_epoch = 1
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 2000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.1
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 1
default.ckpt = 3
default.lr_steps = '100000,160000,220000'
default.models_root = './models'

def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
        config[k] = v
        if k in default:
            default[k] = v
    for k, v in network[_network].items():
        config[k] = v
        if k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        config[k] = v
        if k in default:
            default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset