import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score
from einops import rearrange
from linformer import LinformerSelfAttention
import torch.nn.functional as F


def seg_loss(seg_pred, seg_true, seg_pos_weight=1):
    seg_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(seg_pos_weight))(seg_pred, seg_true)
    return seg_loss


def cls_loss(cls_pred, cls_true, annotated, annotated_weight=1.0, focal=0):
    cls_loss = nn.BCEWithLogitsLoss(reduction='none')(cls_pred, cls_true)
    if focal > 0:
        alpha = 1.
        probas = torch.sigmoid(cls_pred)
        cls_loss = torch.where(cls_true >= 0.5, alpha * (1. - probas) ** focal * cls_loss, probas ** focal * cls_loss)
    cls_loss = cls_loss.mean(dim=1)
    if annotated_weight != 1:
        cls_loss = torch.where(annotated == 1, cls_loss * annotated_weight, cls_loss)
    return cls_loss.mean()


def get3ddots(x):
    return torch.bmm(x.permute(2, 0, 1), x.permute(2, 1, 0)).permute(1, 2, 0)


# def supcon_loss(feats, labels, pos_weight):
#     labels = labels.unsqueeze(1)
#     bcewlogits = nn.BCEWithLogitsLoss(reduction='none')
#     feats = get3ddots(feats)
#     pos_labels = get3ddots(labels)
#     pos_loss = bcewlogits(feats*pos_labels, pos_labels).sum()/pos_labels.sum()
#     neg_labels = get3ddots(1-labels)
#     neg_loss = bcewlogits(feats*neg_labels, neg_labels).sum()/neg_labels.sum()
#     return pos_loss*pos_weight + neg_loss*(1-pos_weight)

# def supcon_loss(features, labels, temperature=0.07):
#
#     labels = labels.unsqueeze(1)
#     mask = get3ddots(labels)+get3ddots(1-labels)
#
#     # compute logits
#     anchor_dot_contrast = torch.div(get3ddots(features), temperature)
#     # for numerical stability
#     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#     logits = anchor_dot_contrast - logits_max.detach()
#
#     # mask-out self-contrast cases
#     logits_mask = (1-torch.diag(torch.ones(features.shape[0])).unsqueeze(-1)).to(features.device)
#     mask = mask * logits_mask
#
#     # compute log_prob
#     logits = logits * logits_mask
#     exp_logits = torch.exp(logits)
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#     # compute mean of log-likelihood over positive
#     if mask.sum() == 0:
#         loss = torch.tensor(0).to(features.device)
#     else:
#         loss = - (mask * log_prob).sum() / mask.sum()
#     return loss


def supcon_loss(features_lst, labels_lst, contrast_mode='all', temperature=0.07, base_temperature=0.07):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    res = []
    device = features_lst.device
    for i in range(11):
        features = features_lst[:, :, i].unsqueeze(1)
        labels = labels_lst[:, i]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if mask.sum() == 0:
            res.append(torch.tensor(0.0).to(device))
        else:
            mean_log_prob_pos = (mask * log_prob).sum() / mask.sum()

            # loss
            loss = - (temperature / base_temperature) * mean_log_prob_pos
            res.append(loss)

    return torch.stack(res).mean()


def kaggle_metric(pred, true):
    score = 0
    for i in range(pred.shape[1]):
        cur_true = true[:, i]
        score += 0.5 if len(np.unique(cur_true)) == 1 else roc_auc_score(cur_true, pred[:, i])
    return score/pred.shape[1]


class Attn(nn.Module):
    def __init__(self, cfg, dim):
        super(Attn, self).__init__()
        seq_len = 144 if cfg.resolution == 768 else 1024
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            LinformerSelfAttention(dim, seq_len, k=cfg.k, heads=cfg.nheads, dropout=cfg.dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.pos_embedding
        x = self.attn(x) + x
        x = rearrange(x, 'b (h w) c ->  b c h w', h=h, w=w)
        return x


class RANZCRModel(nn.Module):
    def __init__(self, cfg):
        super(RANZCRModel, self).__init__()

        self.cfg = cfg
        if cfg.downconv:
            self.avgpool = nn.AvgPool2d(2)
            self.downconv = nn.Sequential(
                nn.Conv2d(1, cfg.in_channels-1, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(cfg.in_channels-1),
                nn.ReLU(),
            )
        if cfg.init_random:
            encoder_weights = None
        else:
            if 'efficientnet' in cfg.encoder:
                encoder_weights = "noisy-student"
            else:
                encoder_weights = 'imagenet'
        self.unet = smp.Unet(
            encoder_name=cfg.encoder,
            encoder_depth=cfg.encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=cfg.in_channels if cfg.downconv else 1,
            classes=11,
            activation=None
        )
        dim = self.unet.encoder._out_channels[-1]
        self.unet.classification_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim, 11),
        )
        if self.cfg.use_attn:
            self.attn = Attn(cfg, dim)
        self.supcon_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 32*11)
        )

    def forward(self, x):
        if self.cfg.downconv:
            x = torch.cat((self.avgpool(x), self.downconv(x)), dim=1)
            resolution = self.cfg.resolution // 2
        else:
            resolution = self.cfg.resolution
        if self.cfg.seg_pretrain:
            features = self.unet.encoder(x)
            seg_pred = self.unet.segmentation_head(self.unet.decoder(*features))
            cls_pred = torch.zeros(x.shape[0], 11).to(x.device)
            supcon_feats = None
        elif self.cfg.supcon:
            cls_pred = torch.zeros(x.shape[0], 11).to(x.device)
            seg_pred = torch.zeros(x.shape[0], 11, resolution, resolution).to(x.device)
            supcon_feats = F.normalize(self.supcon_head(self.unet.encoder(x)[-1].mean(dim=(2, 3))).reshape(x.shape[0], -1, 11), dim=1)
        else:
            last_features = self.unet.encoder(x)[-1]
            if self.cfg.use_attn:
                last_features = self.attn(last_features)
            cls_pred = self.unet.classification_head(last_features.mean(dim=(2, 3)))
            seg_pred = torch.zeros(x.shape[0], 11, resolution, resolution).to(x.device)
            supcon_feats = None
        return cls_pred, seg_pred, supcon_feats