import os
import cv2
import joblib
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from train import CFG
from infer import infer


def preprocess(uid, labels, data_dir):

    res = {}

    # image
    img = cv2.imread(os.path.join(data_dir, f'{uid}.jpg'), cv2.IMREAD_GRAYSCALE)
    res['img'] = img

    res['labels'] = labels
    res['masks'] = []
    res['tips'] = []
    res['annotated'] = 0
    res['mask_exist_index'] = []

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
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataparallel', type=int, default=1)
    args = parser.parse_args()

    cfg = CFG(vars(args))
    os.makedirs(args.save_dir, exist_ok=True)
    uids, preds = infer(cfg)

    def save_one_data(idx):
        uid = uids[idx]
        res = preprocess(uid, preds[idx], args.data_dir)
        joblib.dump(res, os.path.join(args.save_dir, f'{uid}.jl'), compress=args.compress)

    with Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(save_one_data, range(len(uids))), total=len(uids)))
