from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import albumentations as albu
import os
import joblib
import torch


def center_crop(img, ratio):
    if ratio == 1:
        return img
    h, w = img.shape
    crop_h, crop_w = int(h * ratio), int(w * ratio)
    upper_margin, left_margin = (h - crop_h) // 2, (w - crop_w) // 2
    return img[upper_margin:upper_margin + crop_h, left_margin:left_margin + crop_w]


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
        self.mean, self.std = 0.482288, 0.22085
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
                albu.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transforms = albu.Compose([
                albu.Resize(cfg.resolution, cfg.resolution),
                albu.Normalize(mean=self.mean, std=self.std)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.mode == 'test':
            data = {'img': cv2.imread(path, cv2.IMREAD_GRAYSCALE)}
            mask = data['img'] > 0
            data['img'] = data['img'][np.ix_(mask.any(1), mask.any(0))]
            if self.cfg.normalize:
                data['img'] = cv2.normalize(data['img'], None, 0, 255, cv2.NORM_MINMAX)
            data['img'] = center_crop(data['img'], self.cfg.centercrop)
            data['img'] = self.transforms(image=data['img'])['image']
        else:
            data = joblib.load(path)
            mask = data['img'] > 0
            data['img'] = data['img'][np.ix_(mask.any(1), mask.any(0))]
            if self.cfg.normalize:
                data['img'] = cv2.normalize(data['img'], None, 0, 255, cv2.NORM_MINMAX)
            data['img'] = center_crop(data['img'], self.cfg.centercrop)
            # because we downsample image in the network
            mask_size = (self.cfg.resolution//2, self.cfg.resolution//2) if self.cfg.downconv else (self.cfg.resolution, self.cfg.resolution)
            if self.cfg.seg_pretrain:
                if data['annotated'] == 1:
                    exist_index = data['mask_exist_index']
                    transformed = self.transforms(image=data['img'], masks=[center_crop(x, self.cfg.centercrop) for x in data['masks']])
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
            if self.cfg.black_white:
                data['masks'] = (data['img'] > data['img'].mean() * 0.8).astype(np.float32)
                if self.cfg.downconv:
                    data['masks'] = cv2.resize(data['masks'], mask_size)
                data['masks'] = np.expand_dims(data['masks'], 0)
        # datatype
        out = dict()
        for k in ['img', 'masks', 'annotated', 'labels']:
            if k in data.keys():
                out[k] = torch.tensor(data[k], dtype=torch.float)
        out['img'] = out['img'].unsqueeze(0)

        return out


class SIMSIAMDataset(Dataset):
    def __init__(self, cfg):
        base = '../../input/processed/nihchest'
        paths = os.listdir(base)
        self.cfg = cfg
        self.paths = [os.path.join(base, x) for x in paths]
        self.mean, self.std = 0.482288, 0.22085
        self.transforms = albu.Compose([
            albu.RandomResizedCrop(cfg.resolution, cfg.resolution, scale=(0.5, 1), p=1),
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
            # albu.OneOf([
            #     albu.OpticalDistortion(distort_limit=1.0),
            #     albu.GridDistortion(num_steps=5, distort_limit=1.),
            #     albu.ElasticTransform(alpha=3),
            # ], p=0.7),
            albu.CLAHE(clip_limit=4.0, p=0.7),
            albu.IAAPiecewiseAffine(p=0.2),
            albu.IAASharpen(p=0.2),
            albu.RandomGamma(gamma_limit=(70, 130), p=0.3),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            # albu.OneOf([
            #     albu.ImageCompression(),
            #     albu.Downscale(scale_min=0.7, scale_max=0.95),
            # ], p=0.2),
            albu.CoarseDropout(max_holes=8, max_height=int(cfg.resolution * 0.1),
                               max_width=int(cfg.resolution * 0.1), p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albu.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = joblib.load(path)['img']
        if self.cfg.normalize:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = center_crop(image, self.cfg.centercrop)
        img1, img2 = self.transforms(image=image)['image'], self.transforms(image=image)['image']
        img1, img2 = np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)
        return {'img1': img1, 'img2': img2}


def get_loader(paths, cfg, mode='train', distributed=True):
    dataset = SIMSIAMDataset(cfg) if cfg.simsiam else RANZCRDataset(paths, cfg, mode=mode)
    if distributed:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                            sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=mode == 'train'))
    else:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=mode == 'train', pin_memory=True)
    return loader
