import torch
import numpy as np
import pandas as pd
import argparse
import joblib
import os
from tqdm import tqdm

from model import RANZCRModel, kaggle_metric
from data import get_loader
from train import CFG
from utils import sigmoid


def infer(cfg):
    # define device
    device = 'cpu' if (cfg.cpu or (not torch.cuda.is_available())) else 'cuda'
    print(device)

    # load models
    models = []
    fnames = os.listdir(cfg.model_dir)
    for fname in fnames:
        ckp = torch.load(os.path.join(cfg.model_dir, fname), map_location='cpu')
        print('')
        print(fname)
        ckp['cfg']['init_random'] = True
        print(ckp['cfg'])
        model = RANZCRModel(CFG(ckp['cfg']))
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(model.load_state_dict(ckp['model']))
        model.to(device)
        model.eval()
        if cfg.dataparallel:
            model = torch.nn.DataParallel(model)
        models.append(model)
        print(ckp['val_metric'])
        print('')

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
    weights = cfg.weights.split('-')
    preds = []
    with torch.no_grad():
        for data in tqdm(loader):
            batch_preds = []
            for model, weight in zip(models, weights):
                x = data['img'].to(device)
                orig_cls_pred, orig_seg_pred, supcon_feats = model(x)
                # flipped_cls_pred, flipped_seg_pred = model(x.flip(-1))
                # cls_pred = (orig_cls_pred + flipped_cls_pred) / 2
                # batch_preds.append(cls_pred.cpu().numpy())
                batch_preds.append(orig_cls_pred*float(weight))
            batch_preds = torch.sum(torch.stack(batch_preds, dim=0), dim=0)
            preds.append(batch_preds)
    preds = torch.sigmoid(torch.cat(preds, dim=0)).cpu().numpy()
    return uids, preds


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--folds_path', type=str, default='folds.jl')
    parser.add_argument('--train_df_path', type=str, default='../../input/kaggle/train.csv')
    parser.add_argument('--dataparallel', type=int, default=0)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    cfg = CFG(vars(args))

    uids, preds = infer(cfg)

    # make submission csv
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    submission = pd.DataFrame(preds, index=uids, columns=label_cols)
    submission.index = submission.index.rename('StudyInstanceUID')
    submission.to_csv(os.path.join(args.out_dir, 'submission.csv'), index=True)

    if args.fold is not None:
        train_df = pd.read_csv(args.train_df_path, index_col='StudyInstanceUID')
        print(f'Fold{args.fold} SCORE', kaggle_metric(submission.values, train_df.loc[submission.index, label_cols].values))
