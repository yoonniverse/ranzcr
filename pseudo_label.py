import os
import cv2
import joblib
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from train import CFG
from infer import infer
import numpy as np


def preprocess(uid, labels, data_dir):

    # image
    res = {'img': cv2.imread(os.path.join(data_dir, uid + '.jpg'), cv2.IMREAD_GRAYSCALE), 'labels': labels, 'masks': [],
           'tips': [], 'annotated': 0, 'mask_exist_index': []}
    if res['img'].shape[0] * res['img'].shape[1] > 2048*2048:
        res['img'] = cv2.resize(res['img'], (2048, 2048))
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--compress', type=int)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataparallel', type=int, default=1)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--tta', type=int, default=0)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--save_workers', type=int)
    args = parser.parse_args()

    cfg = CFG(vars(args))
    os.makedirs(args.save_dir, exist_ok=True)

    if os.path.exists(f'{args.save_name}.jl'):
        uids, preds = joblib.load(f'{args.save_name}.jl')
    else:
        uids, preds = infer(cfg)
        joblib.dump((uids, preds), f'{args.save_name}.jl')

    rel_index = np.where(preds.max(axis=1) > 0.5)[0]
    print('# process images:', len(rel_index), '/', len(uids))
    uids = np.array(uids)[rel_index]
    preds = preds[rel_index]


    def save_one_data(idx):
        uid = uids[idx]
        res = preprocess(uid, preds[idx], args.data_dir)
        joblib.dump(res, os.path.join(args.save_dir, f'{uid}.jl'), compress=args.compress)

    with Pool(args.save_workers) as pool:
        list(tqdm(pool.imap_unordered(save_one_data, range(len(uids))), total=len(uids)))
