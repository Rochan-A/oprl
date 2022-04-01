import argparse
import yaml
from easydict import EasyDict

import os
import os.path as osp
from os import listdir
from os.path import isfile, join

from pprint import pprint
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

import gym
from gym.wrappers import RescaleAction

import pdb
from torch.utils.tensorboard import SummaryWriter

from agents import TOPAgent
from learners import top_trainer as train



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        help="Path to Config file",
        required=True
    )
    parser.add_argument(
        "--cuda-device",
        help="CUDA device to use",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        help="set numpy & torch seed",
        type=int,
        default=0
    )

    args = parser.parse_args()
    device = "cuda:{}".format(args.cuda_device) if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    # init env
    env = gym.make(config.env)
    env = RescaleAction(env, -1, 1)

    agent = TOPAgent(config, env)
    train(agent, env, args, config)
