#!/usr/bin/env python
# coding: utf-8

# # PPO for Portfolio Management
# This tutorial is to demonstrate an example of using PPO to do portfolio management

# ## Step1: Import Packages

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch

ROOT = os.path.dirname(os.path.abspath("."))
sys.path.append(ROOT)

import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.utils import plot
from trademaster.utils import set_seed
set_seed(2023)


# ## Step2: Import Configs

# In[ ]:


parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
parser.add_argument("--config", default=osp.join(ROOT, "configs", "portfolio_management", "portfolio_management_exchange_ppo_ppo_adam_mse.py"),
                    help="download datasets config file path")
parser.add_argument("--task_name", type=str, default="train")
args, _ = parser.parse_known_args()

cfg = Config.fromfile(args.config)
task_name = args.task_name
cfg = replace_cfg_vals(cfg)


# ## Step3: Build Dataset

# In[ ]:


dataset = build_dataset(cfg)


# ## Step4: Build Trainer

# In[ ]:


from ray.tune.registry import register_env
import ray
from trademaster.environments.portfolio_management.environment import PortfolioManagementEnvironment
def env_creator(env_name):
    if env_name == 'portfolio_management':
        env = PortfolioManagementEnvironment
    else:
        raise NotImplementedError
    return env
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
work_dir = os.path.join(ROOT, cfg.trainer.work_dir)
ray.init(ignore_reinit_error=True)
register_env("portfolio_management", lambda config: env_creator("portfolio_management")(config))
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
cfg.dump(osp.join(work_dir, osp.basename(args.config)))

trainer = build_trainer(cfg, default_args=dict(dataset=dataset, device = device))


# ## Step5: Train, Valid and Test

# In[ ]:


trainer.train_and_valid()


# rllib uses distributed training stategy and therefore bad result migh occurs during training

# In[ ]:


import ray
from ray.tune.registry import register_env
from trademaster.environments.portfolio_management.environment import PortfolioManagementEnvironment
def env_creator(env_name):
    if env_name == 'portfolio_management':
        env = PortfolioManagementEnvironment
    else:
        raise NotImplementedError
    return env
ray.init(ignore_reinit_error=True)
register_env("portfolio_management", lambda config: env_creator("portfolio_management")(config))
trainer.test();


# In[ ]:


plot(trainer.test_environment.save_asset_memory(),alg="PPO")

