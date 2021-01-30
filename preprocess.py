import numpy as np
import pandas as pd
import os
import cv2
import joblib
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import argparse


def preprocess(uid, labels, df_annotations, data_dir):

    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    res = {}

    # image
    img = cv2.imread(os.path.join(os.path.join(data_dir, 'train'), f'{uid}.jpg'), cv2.IMREAD_GRAYSCALE)
    res['img'] = img

    # labels
    res['labels'] = labels

    # masks
    height, width = img.shape[0], img.shape[1]
    res['masks'] = []
    res['tips'] = []
    res['annotated'] = 0
    res['mask_exist_index'] = []
    if uid in df_annotations.index:
        res['annotated'] = 1
        tmp = df_annotations.loc[[uid]]
        labels = tmp['label'].tolist()
        points_lst = tmp['data'].apply(eval).tolist()

        for label, points in zip(labels, points_lst):
            lab_idx = label_cols.index(label)
            res['mask_exist_index'].append(lab_idx)
            mask = np.zeros((height, width), dtype=np.int8)
            for i in range(len(points) - 1):
                start, end = points[i], points[i + 1]
                x_start = start[0]
                x_end = end[0]
                y_start = start[1]
                y_end = end[1]
                if x_end != x_start:
                    slope = (y_end - y_start) / (x_end - x_start)
                    for j in range(min(x_start, x_end), max(x_start, x_end)):
                        mask[np.clip(int(y_start + slope * (j - x_start)), 0, height - 1), max(0, j - 5):min(j + 6, width)] = 1
                if y_end != y_start:
                    slope = (x_end - x_start) / (y_end - y_start)
                    for j in range(min(y_start, y_end), max(y_start, y_end)):
                        mask[max(0, j - 5):min(j + 6, height), np.clip(int(x_start + slope * (j - y_start)), 0, width - 1)] = 1
            res['masks'].append(mask)
            mid_h, mid_w = height/2, width/2
            euc_dist = lambda x: np.sqrt((x[0]-mid_h)**2 + (x[1]-mid_w)**2)
            tip = points[-1] if euc_dist(points[0]) > euc_dist(points[-1]) else points[0]
            res['tips'].append(tip)
    res['masks'] = np.array(res['masks'], dtype=np.int8)
    res['tips'] = np.array(res['tips'], dtype=np.int16)

    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--njobs', type=int)
    parser.add_argument('--nfolds', type=int)
    parser.add_argument('--compress', type=int)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]

    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df_annotations = pd.read_csv(os.path.join(args.data_dir, 'train_annotations.csv'), index_col='StudyInstanceUID')
    uids = df['StudyInstanceUID'].values
    labels = df[label_cols].values

    gkf = GroupKFold(args.nfolds)
    folds = []
    tmp = list(gkf.split(df, df[label_cols], df['PatientID']))
    for trn_idx, val_idx in tmp:
        np.random.seed(0)
        np.random.shuffle(trn_idx)
        np.random.seed(0)
        np.random.shuffle(val_idx)
        folds.append((uids[trn_idx], uids[val_idx]))
    joblib.dump(folds, 'folds.jl')

    def save_one_data(idx):
        uid = uids[idx]
        res = preprocess(uid, labels[idx], df_annotations, args.data_dir)
        joblib.dump(res, os.path.join(args.save_dir, f'{uid}.jl'), compress=args.compress)

    with Pool(args.njobs) as pool:
        list(tqdm(pool.imap_unordered(save_one_data, range(len(uids))), total=len(uids)))
