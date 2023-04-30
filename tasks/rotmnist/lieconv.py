from datetime import datetime

import dill
import torch
import wandb
from torch.utils.data import DataLoader

from consts import PROJ_NAME, ENT_NAME, DEVICE
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config
import copy
import architectures.LieConv.lie_conv.lieGroups as lieGroups
import architectures.LieConv.lie_conv.lieConv as lieConv
from architectures.LieConv.lie_conv.datasets import MnistRotDataset
from architectures.baselines import GIGPImgLieResnet


def makeTrainer(*, dataset=MnistRotDataset, network=GIGPImgLieResnet, num_epochs=100,
                bs=50, lr=3e-3, aug=True, optim=Adam, device=DEVICE, trainer=Classifier,
                split={'train': 12000}, small_test=False, net_config={}, opt_config={},
                trainer_config={'log_dir': None}, checkpoint: str = None):
    print(f'GIGP:{net_config["gigp"]}')

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'), splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)

    if 'test' in split:
        datasets['test'].images = datasets['test'].images[:, :split['test']]
        datasets['test'].labels = datasets['test'].labels[:split['test']]
        datasets['test'].num_samples = split['test']

    model = network(num_targets=datasets['train'].num_targets, **net_config).to(DEVICE)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(), model)
    model, bs = try_multigpu_parallelize(model, bs)

    if checkpoint:
        with open(checkpoint, 'rb') as f:
            checkpoint_data = dill.load(f)

        model.load_state_dict(checkpoint_data['model_state'], strict=False)

    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=bs, shuffle=(k == 'train'),
                                          num_workers=0, pin_memory=False), device) for k, v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'], 1 + len(dataloaders['train']) // 10)
    if small_test: dataloaders['test'] = islice(dataloaders['test'], 1 + len(dataloaders['train']) // 10)
    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)

    wandb.login()

    run_time = datetime.now()
    wandb.init(
        project=PROJ_NAME,
        entity=ENT_NAME,
        name=f'LC_{trainer_config["log_suffix"]}_{"no_" * (not net_config["gigp"])}gigp_{run_time.month}_{run_time.day}_{run_time.hour}_{run_time.minute}_{run_time.second}',
        config={**trainer_config, **net_config})

    return trainer(model, dataloaders, opt_constr, lr_sched, **trainer_config)


"""
python gigp/tasks/rotmnist/lieconv.py --num_epochs=500 --trainer_config "{'log_suffix':'mnistSO2'}" --net_config "{'k':128,'total_ds':.1,'fill':.1,'nbhd':25,'group':SO2(.2), 'gigp': True, 'use_orbits_data': True, 'orbs_agg_dist': .5, 'gigp_agg': 'weighted_sum'}" --bs 25 --lr .003 --split "{'train':10000, 'test':2000}" --aug=True
"""
if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save'] = True
    res = Trial(argupdated_config(defaults, namespace=(lieConv, lieGroups)))
