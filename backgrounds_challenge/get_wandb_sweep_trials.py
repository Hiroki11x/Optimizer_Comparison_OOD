import glob
import wandb

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--optimizer', default=None)
parser.add_argument('--sweep-id', default=None)
args = parser.parse_args()

api = wandb.Api()
optimizer = args.optimizer
sweep_id = args.sweep_id
sweep = api.sweep(f"entity_name/sweep_project_{optimizer}/{sweep_id}") # /entity_name/sweep_project_adam
runs = sorted(
    sweep.runs, 
    key=lambda run: run.summary.get("val_accuracy", 0), 
    reverse=True
)

for run in runs:
    if run.state != 'crashed':
        print(run.name)
