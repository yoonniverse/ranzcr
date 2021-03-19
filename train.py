import torch
import numpy as np
import pandas as pd
import argparse
import joblib
import os
import warnings
from time import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from model import RANZCRModel, EncoderModel, SIMSIAMWrapper, seg_loss, cls_loss, supcon_loss, kaggle_metric, two_view_to_standard
from data import get_loader
from utils import make_reproducible, mkdir, sigmoid
import torch.distributed as dist

warnings.filterwarnings('ignore')


class CFG:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            self.__setattr__(k, v)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:3456', rank=rank, world_size=world_size)


def main_worker(rank, world_size, cfg):
    print(f'Launched gpu {rank}')
    setup(rank, world_size)
    cfg.batch_size = cfg.batch_size // world_size
    cfg.num_workers = cfg.num_workers // world_size

    # for reproduciblity
    make_reproducible(42) if cfg.resume else make_reproducible(cfg.seed)

    # create logdir
    mkdir(cfg.logdir)

    # load data
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    tube_index_dict = {'ett': [0, 1, 2], 'ngt': [3, 4, 5, 6], 'cvc': [7, 8, 9], 'swan': [10]}
    folds = joblib.load('folds.jl')
    train_uids, valid_uids = folds[cfg.fold][0], folds[cfg.fold][1]
    train_uids = train_uids[:int(cfg.data_frac * len(train_uids))]
    if cfg.seg_pretrain:
        annotations_df = pd.read_csv(os.path.join(cfg.df_dir, 'train_annotations.csv'), index_col='StudyInstanceUID')
        annotations_index = annotations_df.index
        train_uids = list(set(annotations_index) & set(train_uids))
        valid_uids = list(set(annotations_index) & set(valid_uids))
    train_paths = [os.path.join(cfg.data_dir, uid) + '.jl' for uid in train_uids]
    valid_paths = [os.path.join(cfg.data_dir, uid) + '.jl' for uid in valid_uids]
    if cfg.pseudo_data_dir:
        pseudo_paths = os.listdir(cfg.pseudo_data_dir)
        pseudo_paths = [os.path.join(cfg.pseudo_data_dir, path) for path in pseudo_paths]
        train_paths += pseudo_paths
    if cfg.full:
        train_paths += valid_paths
    trn_loader = get_loader(train_paths, cfg, mode='train')
    val_loader = get_loader(valid_paths, cfg, mode='valid')
    if rank == 0: print(f'# Train Samples: {len(trn_loader.dataset)} | # Val Samples: {len(val_loader.dataset)}')

    load_simsiam_wrapper = False
    extract_root = False
    if cfg.simsiam:
        load_simsiam_wrapper = True
    if cfg.pretrained_path is not None:
        if cfg.from_simsiam:
            load_simsiam_wrapper = True
            extract_root = True
    model = EncoderModel(cfg) if cfg.encoder_only else RANZCRModel(cfg)
    if load_simsiam_wrapper:
        model = SIMSIAMWrapper(model, cfg)
    model = model.to(rank)

    # define optimizer
    kwargs = []
    max_lrs = []
    named_parameters = model.root.named_parameters() if extract_root else model.named_parameters()
    for name, params in named_parameters:
        tmp = {'params': params, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay}
        if any([x in name for x in ['bias', 'bn']]):
            tmp['weight_decay'] = 0
        if ('projection_mlp' in name) or ('prediction_mlp' in name):
            tmp['lr'] = 0.05 * cfg.batch_size*world_size/256
        if 'decoder' in name:
            tmp['lr'] = cfg.lr * cfg.decoder_lr_ratio
        kwargs.append(tmp)
        max_lrs.append(tmp['lr'])
    if cfg.simsiam:
        optimizer = torch.optim.SGD(kwargs, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(kwargs, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # define lr scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, pct_start=0.01, max_lr=max_lrs, steps_per_epoch=len(trn_loader), epochs=cfg.epochs,
        final_div_factor=1e1 if cfg.focal else 1e4
    )

    # mixed precision
    amp_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # pretrained
    if cfg.pretrained_path is not None:
        checkpoint = torch.load(cfg.pretrained_path, map_location='cpu')
        _ = model.load_state_dict(checkpoint['model'], strict=False)
        if extract_root: model = model.root
        if rank == 0:
            print(_)
            print(f'Loaded Pretrained Weights from {cfg.pretrained_path}')

    n_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0: print(f'# Parameters: {n_parameters}')

    if cfg.resume:
        checkpoint = torch.load(os.path.join(cfg.logdir, 'last.pth'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if os.path.isfile(os.path.join(cfg.logdir, 'best.pth')):
            best_val_metric = torch.load(os.path.join(cfg.logdir, 'best.pth'), map_location='cpu')['val_metric']
        else:
            best_val_metric = -np.inf
        amp_scaler.load_state_dict(checkpoint['amp_scaler'])
        if rank == 0: print(f'Resume from {cfg.logdir}/last.pth. Best val metric: {best_val_metric}')
    else:
        start_epoch = 0
        best_val_metric = -np.inf

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    t0 = time()
    # loop
    for epoch in range(start_epoch, cfg.epochs):

        t00 = time()

        # check current learning rates
        current_lrs = [x["lr"] for x in optimizer.param_groups]
        if rank == 0: print(f'EPOCH: {epoch} | LRs: {set(current_lrs)}')

        # train
        model.train()
        train_loss = 0.0
        if rank == 0:
            iterator = tqdm(enumerate(trn_loader), total=len(trn_loader))
        else:
            iterator = enumerate(trn_loader)
        for i, data in iterator:
            bs = None
            for k in data.keys():
                if bs is None: bs = len(data[k])
                data[k] = data[k].to(rank)
            if cfg.catheter:
                data['labels'] = torch.stack([data['labels'][:, tube_index].max(dim=1)[0].contiguous() for tube_index in
                                              tube_index_dict.keys()], dim=1)
            if cfg.ft_tube:
                data['labels'] = data['labels'][:, tube_index_dict[cfg.ft_tube]].contiguous()
                data['masks'] = data['masks'][:, tube_index_dict[cfg.ft_tube]].contiguous()
            if cfg.ce_loss:
                data['labels'] = torch.cat((data['labels'], 1-data['labels'].max(dim=1, keepdim=True)[0]), dim=1).contiguous()

            if cfg.two_view:
                tubes = [data['labels'][:, tube_index].max(dim=1)[0].contiguous() for tube_index in tube_index_dict.keys()]
                abnormal = data['labels'][:, [0, 3, 7]].max(dim=1)[0].contiguous()
                borderline = data['labels'][:, [1, 4, 8]].max(dim=1)[0].contiguous()
                normal = data['labels'][:, [2, 6, 9]].max(dim=1)[0].contiguous()
                incompletely_imaged = data['labels'][:, 5].contiguous()
                data['labels'] = torch.stack(tubes + [abnormal, borderline, normal, incompletely_imaged], dim=1)

            if (cfg.mixup > 0) and (np.random.rand() > 0.5):
                with torch.no_grad():
                    lam = np.random.beta(cfg.mixup, cfg.mixup)
                    index = torch.randperm(data['img'].shape[0]).to(rank)
                    data['img'] = lam * data['img'] + (1 - lam) * data['img'][index]
                    data['labels'] = lam * data['labels'] + (1 - lam) * data['labels'][index]
                    data['masks'] = lam * data['masks'] + (1 - lam) * data['masks'][index]

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                if cfg.simsiam:
                    loss = model(data['img1'], data['img2'])
                else:
                    cls_logits, seg_logits, supcon_feats = model(data['img'])
                    if cfg.ft_tube:
                        output_len = len(tube_index_dict[cfg.ft_tube])
                        if cfg.ce_loss: output_len += 1
                        cls_logits = cls_logits[:, range(output_len)].contiguous()
                        seg_logits = seg_logits[:, range(output_len)].contiguous()
                    if cfg.seg_pretrain or cfg.black_white:
                        loss = seg_loss(seg_logits, data['masks'], seg_pos_weight=cfg.seg_pos_weight, focal=cfg.focal)
                    elif cfg.supcon:
                        loss = supcon_loss(supcon_feats, data['labels'])
                    else:
                        loss = cls_loss(cls_logits, data['labels'], data['annotated'],
                                        annotated_weight=cfg.annotated_weight, focal=cfg.focal, ce=cfg.ce_loss)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            # gather all
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            # g_cls_logits = [torch.zeros(cls_logits.shape).to(rank) for _ in range(world_size)]
            # dist.all_gather(g_cls_logits, cls_logits)
            # g_cls_logits = torch.cat(g_cls_logits, dim=0)
            # g_labels = [torch.zeros(data['labels'].shape).to(rank) for _ in range(world_size)]
            # dist.all_gather(g_labels, data['labels'])
            # g_labels = torch.cat(g_labels, dim=0)
            # add batch loss and metric
            if not torch.isfinite(loss):
                joblib.dump((data, cls_logits, seg_logits, loss, grad_norm), 'error.jl')
                print('LOSS NOT FINITE')
                exit()
            train_loss += loss.item() * bs / (len(trn_loader.dataset))
            # train_metric += kaggle_metric(np.nan_to_num(g_cls_logits.detach().cpu().numpy()),
            #                               g_labels.detach().cpu().numpy()) * len(data['img']) * world_size / len(
            #     trn_loader.dataset)

        str_train_loss = np.round(train_loss, 6)
        if rank == 0:
            print(f'(trn) LOSS: {str_train_loss}', end=' || ')

        # validate
        if (epoch % cfg.val_freq == 0) and (not cfg.full) and (not cfg.simsiam):
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                iterator = tqdm(val_loader) if rank == 0 else val_loader
                for data in iterator:
                    for k in data.keys():
                        data[k] = data[k].to(rank)
                    if cfg.catheter:
                        data['labels'] = torch.stack(
                            [data['labels'][:, tube_index].max(dim=1)[0].contiguous() for tube_index in
                             tube_index_dict.keys()], dim=1)
                    if cfg.ft_tube:
                        data['labels'] = data['labels'][:, tube_index_dict[cfg.ft_tube]].contiguous()
                        data['masks'] = data['masks'][:, tube_index_dict[cfg.ft_tube]].contiguous()
                    cls_logits, seg_logits, supcon_feats = model(data['img'])
                    if cfg.ft_tube:
                        output_len = len(tube_index_dict[cfg.ft_tube])
                        cls_logits = cls_logits[:, range(output_len)].contiguous()
                        seg_logits = seg_logits[:, range(output_len)].contiguous()

                    # gather all
                    g_cls_logits = [torch.zeros(cls_logits.shape).to(rank) for _ in range(world_size)]
                    dist.all_gather(g_cls_logits, cls_logits)
                    g_cls_logits = torch.cat(g_cls_logits, dim=0)
                    g_labels = [torch.zeros(data['labels'].shape).to(rank) for _ in range(world_size)]
                    dist.all_gather(g_labels, data['labels'])
                    g_labels = torch.cat(g_labels, dim=0)

                    # compute loss
                    if cfg.seg_pretrain or cfg.black_white:
                        loss = seg_loss(seg_logits, data['masks'], seg_pos_weight=cfg.seg_pos_weight)
                    elif cfg.supcon:
                        loss = supcon_loss(supcon_feats, data['labels'])
                    else:
                        loss = cls_loss(g_cls_logits, g_labels)

                    # process for two view
                    if cfg.two_view:
                        g_cls_logits = two_view_to_standard(torch.sigmoid(g_cls_logits))

                    # add batch preds, targets, loss
                    val_loss += loss.item() * len(data['img']) * world_size / len(val_loader.dataset)
                    val_preds += g_cls_logits.cpu().tolist()
                    val_targets += g_labels.cpu().tolist()

            # compute validation metric
            val_preds, val_targets = np.array(val_preds), np.array(val_targets)
            if cfg.seg_pretrain or cfg.supcon or cfg.black_white:
                val_metric = -val_loss
            else:
                val_metric = kaggle_metric(val_preds, val_targets)
            str_val_metric = np.round(val_metric, 6)
            str_val_loss = np.round(val_loss, 6)
        else:
            val_loss, val_metric, val_preds, str_val_loss, str_val_metric = None, None, None, None, None

        # add log
        tmp_info = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'amp_scaler': amp_scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'learning_rates': current_lrs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'val_preds': val_preds,
            'cfg': cfg.__dict__
        }
        if val_metric is not None:
            if val_metric > best_val_metric:
                # save info when improved
                best_val_metric = val_metric
                if rank == 0: torch.save(tmp_info, f'{cfg.logdir}/best.pth')
        if rank == 0:
            print(f'(val) LOSS: {str_val_loss} | METRIC: {str_val_metric}', end=' || ')
            with open(f'{cfg.logdir}/log.txt', 'a') as f:
                f.write(
                    f'{epoch} - {str_train_loss} - {str_val_loss} - {str_val_metric} - {set(current_lrs)}\n')

            torch.save(tmp_info, f'{cfg.logdir}/last.pth')
            if epoch % 20 == 19:
                torch.save(tmp_info, f'{cfg.logdir}/e{epoch}.pth')
            print(f'Runtime: {int(time() - t00)}')

    if rank == 0:
        runtime = int(time() - t0)
        print(f'Best Val Score: {best_val_metric}')
        print(f'Runtime: {runtime}')
        # save current argument settings and result to file
        if os.path.exists('history.csv'):
            history = pd.read_csv('history.csv')
        else:
            history = pd.DataFrame(columns=list(cfg.__dict__.keys()))
        cols = history.columns.tolist()

        # rearrange columns
        front_cols = ['logdir', 'score', 'runtime']
        for c in front_cols:
            cols.remove(c)
        cols = front_cols + cols
        history = history[cols]

        info = cfg.__dict__
        info['score'] = best_val_metric
        info['runtime'] = runtime
        info['n_parameters'] = n_parameters
        history = history.append(info, ignore_index=True)
        history.to_csv('history.csv', index=False)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--decoder_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--annotated_weight', type=float, default=1.0)
    parser.add_argument('--seg_pos_weight', type=float, default=100.0)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data_frac', type=float, default=1.)
    parser.add_argument('--amp', type=int, default=1)
    parser.add_argument('--gpu_numbers', type=str, default='0, 1, 2, 3')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--data_dir', type=str, default='../../input/processed/data')
    parser.add_argument('--df_dir', type=str, default='../../input/kaggle/')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--init_random', type=int, default=0)
    parser.add_argument('--use_attn', type=int, default=0)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--focal', type=float, default=0.)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--seg_pretrain', type=int, default=0)
    parser.add_argument('--supcon', type=int, default=0)
    parser.add_argument('--pseudo_data_dir', type=str)
    parser.add_argument('--downconv', type=int, default=1)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--full', type=int, default=0)
    parser.add_argument('--last_conv', type=int, default=0)
    parser.add_argument('--centercrop', type=float, default=1.)
    parser.add_argument('--pos_emb', type=int, default=1)
    parser.add_argument('--mixup', type=float, default=0.)
    parser.add_argument('--catheter', type=int, default=0)
    parser.add_argument('--two_view', type=int, default=0)
    parser.add_argument('--black_white', type=int, default=0)
    parser.add_argument('--ft_tube', type=str)
    parser.add_argument('--ce_loss', type=int, default=0)
    parser.add_argument('--simsiam', type=int, default=0)
    parser.add_argument('--encoder_only', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--from_simsiam', type=int, default=0)
    args = parser.parse_args()
    args.amp = bool(args.amp)
    cfg = CFG(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_numbers
    if cfg.simsiam:
        cfg.weight_decay = 1e-5
    print(cfg.__dict__)
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))
