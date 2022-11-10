from torchvision import transforms
import torch as ch
import torch.nn as nn
import numpy as np
import json
import os
import time
from argparse import ArgumentParser
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import (
    NormalizedModel,
    make_and_restore_model,
    eval_model
)
from pytorch_dnn_arsenal.model import build_model, ModelSetting

# Wandb and Logger
import wandb
import yaml
import glob
        
def load_configs(log_dir, task_name, exp_name) -> dict:
    config_path = os.path.join(log_dir, task_name, exp_name, 'configs.yaml')
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

parser = ArgumentParser()
parser.add_argument('--exp-id', default=None)
parser.add_argument('--trial-id', default=None)
parser.add_argument('--data-path', required=True,
                    help='Path to the eval data')
parser.add_argument('--in9', dest='in9', default=False, action='store_true',
                    help='Enable if the model has 9 output classes, like in IN-9')
parser.add_argument('--set-wandb', default=False, action='store_true')                  

def load_dataloader(variation:str):
    """
    Load eval dataset
    args:
        variation: "mixed_same" or "mixed_rand"
    return:
        validation dataloader
    """
    BASE_PATH_TO_EVAL = args.data_path
    BATCH_SIZE = 32
    WORKERS = 8
    in9_ds = ImageNet9(f'{BASE_PATH_TO_EVAL}/{variation}')
    val_loader = in9_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
    return val_loader

def eval_model_bg(args, variation, ckpt_path):
    map_to_in9 = {}
    with open('in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))
    
    # Load dataset
    train_ds = ImageNet('/path_to_dataset/ImageNet1K/train/')
    val_loader = load_dataloader(variation)

    # build model
    path_prefix = "[checkpoints_path_prefix]"
    project_name = args.exp_id
    trial_id = args.trial_id
    project_dir_path = "{}/{}".format(path_prefix, project_name)
    traj_conf = load_configs(path_prefix, project_name, trial_id)

    model = build_model(
        ModelSetting(
            name=traj_conf['model'],
            num_classes=1000,
            dropout_ratio=traj_conf['dropout_ratio'])
        )
    model.cuda()
    model.eval()
    model = nn.DataParallel(model)

    # Load checkpoint
    print(ckpt_path)
    if not os.path.exists(ckpt_path):
        print("For break is called: {} is not exist".format(model_path))
    check_point = ch.load(ckpt_path)
    model.load_state_dict((check_point))

    # Normalize by dataset mean and std, as is standard.
    model = NormalizedModel(model, train_ds)

    # Evaluate model
    in9_trained = args.in9
    prec1 = eval_model(val_loader, model, map_to_in9, map_in_to_in9=(not in9_trained))
    # print(f'Accuracy on {variation} is {prec1*100:.2f}%')

    return prec1

def main(args):

    path_prefix = "[checkpoints_path_prefix]"
    project_name = args.exp_id
    trial_id = args.trial_id
    project_dir_path = "{}/{}".format(path_prefix, project_name)
    if args.set_wandb:
        traj_conf = load_configs(path_prefix, project_name, trial_id)
        # wandb init
        wandb_configs = {
            'dataset': traj_conf['dataset'],
            'model': traj_conf['model'],
            'optimizer': traj_conf['optimizer'],
            'project_name': project_name,
            'trial_id': trial_id,
            'train_total_batch_size': traj_conf['train_total_batch_size']
        }
        wandb_project_name = "BGC_{}_{}_train_finished".format(traj_conf['optimizer'],traj_conf['model'])
        wandb.init(
            config=wandb_configs, 
            project=wandb_project_name, 
            entity='entity_name', 
            name=trial_id
        )

    ckpt_paths = glob.glob(f'{project_dir_path}/{trial_id}/model*.tar')

    for ckpt_path in ckpt_paths:
    # for ckpt_path in ckpt_paths[-2:-1]:

        acc_orig = eval_model_bg(args, variation='original', ckpt_path=ckpt_path)
        acc_mixed_same = eval_model_bg(args, variation='mixed_same', ckpt_path=ckpt_path)
        acc_mixed_rand = eval_model_bg(args, variation='mixed_rand', ckpt_path=ckpt_path)
        bg_gap = acc_mixed_same*100 - acc_mixed_rand*100
        print(f'Accuracy on original is {acc_orig*100}%')
        print(f'Accuracy on mixed_same is {acc_mixed_same*100}%')
        print(f'Accuracy on mixed_rand is {acc_mixed_rand*100}%')
        print(f'bg gap is {bg_gap}')

        eval_epoch = ckpt_path.split('/')[-1].split('-')[-1].split('.')[0]
        
        if args.set_wandb:
            wandb.log({
                'eval_epoch' : int(eval_epoch),
                'acc_orig' : acc_orig,
                'acc_mixed_same' : acc_mixed_same,
                'acc_mixed_rand' : acc_mixed_rand,
                'bg_gap' : bg_gap
            })

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

