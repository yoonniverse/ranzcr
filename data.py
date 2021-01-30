from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import joblib
import os
import cv2
import albumentations as albu


class RANZCRDataset(Dataset):

    def __init__(self, paths, cfg, mode='train'):
        self.label_cols = [
            'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
            'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
            'Swan Ganz Catheter Present'
        ]
        self.paths = paths
        self.cfg = cfg
        self.mode = mode
        if (self.mode == 'train') and self.cfg.aug:
            self.transforms = albu.Compose([
                albu.RandomResizedCrop(cfg.resolution, cfg.resolution, scale=(0.9, 1), p=1),
                # albu.Resize(cfg.resolution, cfg.resolution),
                # albu.HorizontalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(blur_limit=(3, 5)),
                    albu.MedianBlur(blur_limit=5),
                    albu.GaussianBlur(blur_limit=(3, 5)),
                    albu.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),
                # albu.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
                # albu.OneOf([
                #     albu.GaussNoise(var_limit=(5.0, 40.0), mean=0),
                #     albu.MultiplicativeNoise(multiplier=(0.9, 1.1)),
                # ], p=0.3),
                albu.OneOf([
                    albu.OpticalDistortion(distort_limit=1.0),
                    albu.GridDistortion(num_steps=5, distort_limit=1.),
                    albu.ElasticTransform(alpha=3),
                ], p=0.7),
                albu.CLAHE(clip_limit=4.0, p=0.7),
                albu.IAAPiecewiseAffine(p=0.2),
                albu.IAASharpen(p=0.2),
                albu.RandomGamma(gamma_limit=(70, 130), p=0.3),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                albu.OneOf([
                    albu.ImageCompression(),
                    albu.Downscale(scale_min=0.7, scale_max=0.95),
                ], p=0.2),
                albu.CoarseDropout(max_holes=8, max_height=int(cfg.resolution * 0.1),
                                   max_width=int(cfg.resolution * 0.1), p=0.5),
                albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                albu.Normalize(mean=0.482288, std=0.22085)
            ])
        else:
            self.transforms = albu.Compose([
                albu.Resize(cfg.resolution, cfg.resolution),
                albu.Normalize(mean=0.482288, std=0.22085)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        if self.mode == 'test':
            data = {'img': cv2.imread(path, cv2.IMREAD_GRAYSCALE)}
            data['img'] = self.transforms(image=data['img'])['image']
        else:
            data = joblib.load(path)
            # because we downsample image in the network
            mask_size = (self.cfg.resolution//2, self.cfg.resolution//2) if self.cfg.downconv else (self.cfg.resolution, self.cfg.resolution)
            if self.cfg.seg_pretrain:
                if data['annotated'] == 1:
                    exist_index = data['mask_exist_index']
                    transformed = self.transforms(image=data['img'], masks=[x for x in data['masks']])
                    data['masks'] = np.zeros((11, *mask_size))
                    masks = np.stack([cv2.resize(x.astype(np.float32), mask_size) for x in transformed['masks']], axis=0)
                    data['masks'][exist_index] = masks
                else:
                    transformed = self.transforms(image=data['img'])
                    data['masks'] = np.zeros((11, *mask_size))
            else:
                transformed = self.transforms(image=data['img'])
                data['masks'] = np.zeros((11, *mask_size))
            data['img'] = transformed['image']
        # datatype
        out = dict()
        for k in ['img', 'masks', 'annotated', 'labels']:
            if k in data.keys():
                out[k] = torch.tensor(data[k], dtype=torch.float)
        out['img'] = out['img'].unsqueeze(0)

        return out


def get_loader(uids, cfg, mode='train', distributed=True):
    dataset = RANZCRDataset(uids, cfg, mode=mode)
    if distributed:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                            sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=mode == 'train'))
    else:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=mode == 'train', pin_memory=True)
    return loader
