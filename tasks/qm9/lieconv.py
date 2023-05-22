import dill
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, cosLr, FixedNumpySeed
from oil.tuning.args import argupdated_config
from oil.tuning.study import train_trial
from oil.utils.parallel import try_multigpu_parallelize
from architectures.LieConv.lie_conv.datasets import QM9datasets
from architectures.LieConv.corm_data.collate import collate_fn
from architectures.LieConv.lie_conv.moleculeTrainer import MolecLieResNet, MoleculeTrainer
from architectures.baselines import GIGPMolecLieResNet
from oil.datasetup.datasets import split_dataset

import architectures.LieConv.lie_conv.moleculeTrainer as moleculeTrainer
import architectures.LieConv.lie_conv.lieGroups as lieGroups
import functools
import copy
import logging

logging.basicConfig()
logger = logging.getLogger()


def makeTrainer(
        *,
        task="homo",
        device="cuda",
        lr=3e-3,
        bs=75,
        num_epochs=500,
        network=GIGPMolecLieResNet,
        net_config={
            "k": 1536,
            "nbhd": 100,
            "act": "swish",
            "group": lieGroups.T(3),
            "bn": True,
            "aug": True,
            "mean": True,
            "num_layers": 6,
        },
        recenter=False,
        subsample=False,
        trainer_config={"log_dir": None, "log_suffix": ""},
        checkpoint: str = None,
        disable_base_lc: bool = False
):  # ,'log_args':{'timeFrac':1/4,'minPeriod':0}}):
    # Create Training set and model
    logging.getLogger().setLevel(logging.DEBUG)
    device = torch.device(device)
    with FixedNumpySeed(0):
        logger.info("initialising the QM9 datasets")
        datasets, num_species, charge_scale = QM9datasets()
        if subsample:
            logger.info("subsampling...")
            datasets.update(
                split_dataset(datasets["train"], {"train": subsample})
            )
    ds_stats = datasets["train"].stats[task]
    if recenter:
        logger.info("recentering...")
        m = datasets["train"].data["charges"] > 0
        pos = datasets["train"].data["positions"][m]
        mean, std = pos.mean(dim=0), pos.std()
        for ds in datasets.values():
            ds.data["positions"] = (
                                           ds.data["positions"] - mean[None, None, :]
                                   ) / std

    logger.info("Done preparing the dataset!")
    model = network(num_species, charge_scale, **net_config).to(device)
    model, bs = try_multigpu_parallelize(model, bs)

    if checkpoint:
        print(f'Loading checkpoint {checkpoint}')

        with open(checkpoint, 'rb') as f:
            checkpoint_data = dill.load(f)

        model.load_state_dict(checkpoint_data['model_state'], strict=False)

    if disable_base_lc:
        print('Disabling non-GIGP layers')

        for param_name, param in model.named_parameters():
            if 'gigp' in param_name:
                param.requires_grad_(True)
                continue
            param.requires_grad_(False)

    # Create train and Val(Test) dataloaders and move elems to gpu
    dataloaders = {
        key: LoaderTo(
            DataLoader(
                dataset,
                batch_size=bs,
                num_workers=0,
                shuffle=(key == "train"),
                pin_memory=False,
                collate_fn=collate_fn,
                drop_last=True,
            ),
            device,
        )
        for key, dataset in datasets.items()
    }
    # subsampled training dataloader for faster logging of training performance
    dataloaders["Train"] = islice(
        dataloaders["train"], len(dataloaders["train"]) // 10
    )

    # Initialize optimizer and learning rate schedule
    opt_constr = functools.partial(Adam, lr=lr)
    cos = cosLr(num_epochs)
    lr_sched = lambda e: min(e / (0.01 * num_epochs), 1) * cos(e)
    return MoleculeTrainer(
        model,
        dataloaders,
        opt_constr,
        lr_sched,
        task=task,
        ds_stats=ds_stats,
        **trainer_config
    )


"""--task 'homo' --lr 3e-3 --aug True --num_epochs 500 --num_layers 6 --save=True --net_config "{'group':T(3),'fill':1.,'gigp': True, 'use_orbits_data': True, 'orbs_agg_dist': .5, 'gigp_agg': 'weighted_sum'}" --device cuda --checkpoint "gigp/tasks/qm9/c500.state" --disable_base_lc True"""
if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults["trainer_config"]["early_stop_metric"] = "valid_MAE"
    defaults["save"] = False
    print(
        Trial(
            argupdated_config(defaults, namespace=(moleculeTrainer, lieGroups)),
            use_wandb=False
        )
    )

    # thestudy = Study(simpleTrial,argupdated_config(config_spec,namespace=__init__),
    #                 study_name="point2d",base_log_dir=log_dir)
    # thestudy.run(ordered=False)
