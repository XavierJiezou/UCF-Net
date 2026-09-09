import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yaml
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


parser = argparse.ArgumentParser(description='Evaluate one dataset shard.')
parser.add_argument('--detector_path', type=str, required=True)
parser.add_argument('--test_dataset', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--shard_idx', type=int, required=True)
parser.add_argument('--num_shards', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def load_config():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        test_config = yaml.safe_load(f)
    detector_overrides = dict(config)
    config.update(test_config)
    config.update({k: v for k, v in detector_overrides.items() if v is not None})
    config['test_dataset'] = args.test_dataset
    config['weights_path'] = args.weights_path
    config['workers'] = config.get('workers', 8)
    if torch.cuda.is_available() and "2060" not in torch.cuda.get_device_name():
        config['lmdb_dir'] = r'/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/jikangcheng/data/LMDBs'
    return config


def build_loader(config, dataset):
    indices = np.arange(len(dataset))[args.shard_idx::args.num_shards]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader_kwargs = dict(
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    if int(config['workers']) > 0:
        loader_kwargs['pin_memory'] = bool(config.get('pin_memory', True))
        loader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', True))
        loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 4))
    return torch.utils.data.DataLoader(subset, **loader_kwargs), indices


def load_model(config):
    model = DETECTOR[config['model_name']](config).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    weights = {key.replace('module.', ''): value for key, value in ckpt.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


@torch.no_grad()
def main():
    config = load_config()
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    dataset = DeepfakeAbstractBaseDataset(config=config, mode='test')
    loader, indices = build_loader(config, dataset)
    model = load_model(config)

    preds = []
    labels = []
    for data_dict in tqdm(loader, total=len(loader)):
        label = torch.where(data_dict['label'] != 0, 1, 0)
        data_dict['image'] = data_dict['image'].to(device)
        data_dict['label'] = label.to(device)
        if data_dict.get('mask') is not None:
            data_dict['mask'] = data_dict['mask'].to(device)
        if data_dict.get('landmark') is not None:
            data_dict['landmark'] = data_dict['landmark'].to(device)

        predictions = model(data_dict, inference=True)
        preds.extend(predictions['prob'].cpu().detach().numpy())
        labels.extend(data_dict['label'].cpu().detach().numpy())

    image_names = np.array([dataset.data_dict['image'][i] for i in indices], dtype=object)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        indices=indices,
        pred=np.array(preds),
        label=np.array(labels),
        image=image_names,
    )


if __name__ == '__main__':
    main()
