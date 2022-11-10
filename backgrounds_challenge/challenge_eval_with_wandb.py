from torchvision import transforms
import torch as ch
import torch.nn as nn
import numpy as np
import json
import os
import time
from argparse import ArgumentParser
from PIL import Image
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, adv_bgs_eval_model
from pytorch_dnn_arsenal.model import build_model, ModelSetting

# Wandb and Logger
import wandb
import yaml

def load_configs(log_dir, task_name, exp_name) -> dict:
    config_path = os.path.join(log_dir, task_name, exp_name, 'configs.yaml')
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

parser = ArgumentParser()
parser.add_argument('--exp-id', default=None)
parser.add_argument('--trial-id', default=None)
parser.add_argument('--eval-epoch', default=None,
                    help='epoch of evaluating model.')                    
parser.add_argument('--data-path', required=True,
                    help='Path to the eval data')
parser.add_argument('--in9', dest='in9', default=False, action='store_true',
                    help='Enable if the model has 9 output classes, like in IN-9')

def main(args):
    map_to_in9 = {}
    with open('in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    BASE_PATH_TO_EVAL = args.data_path
    BATCH_SIZE = 32
    WORKERS = 8


    # wandb init
    path_prefix = "[checkpoints_path_prefix]"
    project_name = args.exp_id
    trial_id = args.trial_id
    project_dir_path = "{}/{}".format(path_prefix, project_name)
    traj_conf = load_configs(path_prefix, project_name, trial_id)

    wandb_configs = {
        'dataset': traj_conf['dataset'],
        'model': traj_conf['model'],
        'optimizer': traj_conf['optimizer'],
        'project_name': project_name,
        'trial_id': trial_id,
        'train_total_batch_size': traj_conf['train_total_batch_size']
    }
    
    wandb_project_name = "BGC_{}_{}".format(traj_conf['optimizer'],traj_conf['model'])
    wandb.init(config=wandb_configs, 
               project=wandb_project_name, 
               entity='entity_name', 
               name=trial_id)

    # Load model
    in9_trained = args.in9
    
    model = build_model(
        ModelSetting(name=traj_conf['model'],
                num_classes=1000,
                dropout_ratio=traj_conf['dropout_ratio']))

    model.cuda()
    model = nn.DataParallel(model)

    check_point_name = "model-checkpoint-{}.pth.tar".format(args.eval_epoch)
    model_path = "{}/{}/{}".format(project_dir_path, trial_id, check_point_name)

    print(model_path)
    if not os.path.exists(model_path):
        print("For break is called: {} is not exist".format(model_path))
    check_point = ch.load(model_path)
    model.load_state_dict((check_point))

    model.eval()
    
    # Load backgrounds
    bg_ds = ImageNet9(f'{BASE_PATH_TO_EVAL}/only_bg_t')
    bg_loader = bg_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)

    # Load foregrounds
    fg_mask_base = f'{BASE_PATH_TO_EVAL}/fg_mask/val'
    class_names = sorted(os.listdir(f'{fg_mask_base}'))
    def get_fgs(classnum):
        classname = class_names[classnum]
        return sorted(os.listdir(f'{fg_mask_base}/{classname}'))

    total_vulnerable = 0
    total_computed = 0
    # Big loop
    for fg_class in range(9):

        fgs = get_fgs(fg_class)
        fg_classname = class_names[fg_class]

        # Evaluate model
        prev_time = time.time()
        for i in range(len(fgs)):
            if total_computed % 50 == 0:
                cur_time = time.time()
                print(f'At image {i} for class {fg_classname}, used {(cur_time-prev_time):.2f} since the last print statement.')
                print(f'Up until now, have {total_vulnerable}/{total_computed} vulnerable foregrounds.')
                prev_time = cur_time

            mask_name = fgs[i]
            fg_mask_path = f'{fg_mask_base}/{fg_classname}/{mask_name}'
            fg_mask = np.load(fg_mask_path)
            fg_mask = np.tile(fg_mask[:, :, np.newaxis], [1, 1, 3]).astype('uint8')
            fg_mask = transforms.ToTensor()(Image.fromarray(fg_mask*255))
            
            img_name = mask_name.replace('npy', 'JPEG')
            image_path = f'{BASE_PATH_TO_EVAL}/original/val/{fg_classname}/{img_name}'
            img = transforms.ToTensor()(Image.open(image_path))

            is_adv = adv_bgs_eval_model(bg_loader, model, img, fg_mask, fg_class, BATCH_SIZE, map_to_in9, map_in_to_in9=(not in9_trained))
            # print(f'Image {i} of class {fg_classname} is {is_adv}.')
            total_vulnerable += is_adv
            total_computed += 1

    print('Evaluation complete')
    percent_vulnerable = total_vulnerable/total_computed * 100
    print(f'Summary: {total_vulnerable}/{total_computed} ({percent_vulnerable:.2f}%) are vulnerable foregrounds.')

    wandb.log({'total_vulnerable' : total_vulnerable,
               'total_computed' : total_computed,
               'percent_vulnerable' : percent_vulnerable})

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
