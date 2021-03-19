import torch
import numpy as np
import pandas as pd
import argparse
import joblib
import os
from tqdm import tqdm
from torchvision import transforms

from model import RANZCRModel, kaggle_metric
from data import get_loader
from train import CFG
from utils import sigmoid, mkdir


def infer(cfg):

    tube_index_dict = {'ett': [0, 1, 2], 'ngt': [3, 4, 5, 6], 'cvc': [7, 8, 9], 'swan': [10]}

    # define device
    device = 'cpu' if (cfg.cpu or (not torch.cuda.is_available())) else 'cuda'
    print(device)

    # load models
    models = []
    model_cfgs = []
    fnames = sorted(os.listdir(cfg.model_dir))
    fnames = [x for x in fnames if x.endswith('.pth')]
    if cfg.fname_include:
        fnames = [x for x in fnames if cfg.fname_include in x]
    for j, fname in enumerate(fnames):
        ckp = torch.load(os.path.join(cfg.model_dir, fname), map_location='cpu')
        print('')
        print(fname)
        ckp['cfg']['init_random'] = True
        print(ckp['cfg'])
        model = RANZCRModel(CFG(ckp['cfg']))
        print(model.load_state_dict(ckp['model']))
        model.to(device)
        model.eval()
        if cfg.dataparallel:
            model = torch.nn.DataParallel(model)
        models.append(model)
        model_cfgs.append(CFG(ckp['cfg']))
        print(ckp['val_metric'])
        print('')
        if j == 0:
            existing_keys = cfg.__dict__.keys()
            for k, v in ckp['cfg'].items():
                if k not in existing_keys:
                    cfg.__setattr__(k, v)

    # create loader
    if cfg.fold is None:
        fnames = os.listdir(cfg.data_dir)
        uids = ['.'.join(x.split('.')[:-1]) for x in fnames]
        paths = [os.path.join(cfg.data_dir, x) for x in fnames]
    else:
        folds = joblib.load(cfg.folds_path)
        uids = folds[cfg.fold][1]
        paths = [os.path.join(cfg.data_dir, x+'.jpg') for x in uids]
    print(f'# uids: {len(paths)}')
    loader = get_loader(paths, cfg, mode='test', distributed=False)

    # infer
    if cfg.weights is None:
        weights = np.ones(len(models))
    else:
        weights = np.array(cfg.weights.split('-'), dtype=np.float32)
    weights = weights / weights.sum()
    print('Ensemble weights:', weights)
    tta_fs = [transforms.Compose([])]
    if cfg.tta:
        tta_fs += [
            transforms.Compose([transforms.CenterCrop(int(cfg.resolution*0.95)), transforms.Resize(cfg.resolution)]),
            transforms.Compose([transforms.CenterCrop(int(cfg.resolution*0.9)), transforms.Resize(cfg.resolution)]),
        ]
    preds = []
    with torch.no_grad():
        for data in tqdm(loader):
            batch_preds = []
            for model, model_cfg, weight in zip(models, model_cfgs, weights):
                x = data['img'].to(device)
                b = x.shape[0]
                x = torch.cat([tta_f(x) for tta_f in tta_fs], dim=0)
                orig_cls_pred, orig_seg_pred, supcon_feats = model(x)
                orig_cls_pred = torch.sigmoid(orig_cls_pred).reshape(len(tta_fs), b, 11).mean(dim=0)
                batch_preds.append((orig_cls_pred**cfg.temperature)*weight)
            batch_preds = torch.sum(torch.stack(batch_preds, dim=0), dim=0)
            preds.append(batch_preds)
    preds = torch.cat(preds, dim=0).cpu().numpy()
    if cfg.postprocess:
        new_sub = pd.DataFrame(preds)
        ett_sum = new_sub[tube_index_dict['ett']].sum(axis=1)
        ett_fix_index = new_sub[ett_sum > 0.5].index
        new_sub.loc[ett_fix_index, tube_index_dict['ett']] /= np.expand_dims(ett_sum.loc[ett_fix_index].values, 1)
        ngt_sum = new_sub[tube_index_dict['ngt']].sum(axis=1)
        ngt_fix_index = new_sub[ngt_sum > 0.8].index
        new_sub.loc[ngt_fix_index, tube_index_dict['ngt']] /= np.expand_dims(ngt_sum.loc[ngt_fix_index].values, 1)
        preds = new_sub.values
    return uids, preds


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--out_path', type=str, default='submission.csv')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--folds_path', type=str, default='folds.jl')
    parser.add_argument('--train_df_path', type=str, default='../../input/kaggle/train.csv')
    parser.add_argument('--dataparallel', type=int, default=0)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--fname_include', type=str)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--postprocess', type=int, default=0)
    args = parser.parse_args()

    cfg = CFG(vars(args))
    mkdir('/'.join(cfg.out_path.split('/')[:-1]))

    uids, preds = infer(cfg)

    # make submission csv
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    submission = pd.DataFrame(preds, index=uids, columns=label_cols)
    submission.index = submission.index.rename('StudyInstanceUID')
    submission.to_csv(args.out_path, index=True)

    if args.fold is not None:
        train_df = pd.read_csv(args.train_df_path, index_col='StudyInstanceUID')
        print(f'Fold{args.fold} SCORE', kaggle_metric(submission.values, train_df.loc[submission.index, label_cols].values))
