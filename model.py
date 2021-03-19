import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score
from einops import rearrange
from linformer import LinformerSelfAttention
import torch.nn.functional as F
import timm
from swav.src.resnet50 import MultiPrototypes


def two_view_to_standard(probs):
    tmp = torch.zeros((len(probs), 11)).to(probs.device)
    tmp[:, 0] = probs[:, 0] * probs[:, 4]
    tmp[:, 1] = probs[:, 0] * probs[:, 5]
    tmp[:, 2] = probs[:, 0] * probs[:, 6]
    tmp[:, 3] = probs[:, 1] * probs[:, 4]
    tmp[:, 4] = probs[:, 1] * probs[:, 5]
    tmp[:, 5] = probs[:, 1] * probs[:, 7]
    tmp[:, 6] = probs[:, 1] * probs[:, 6]
    tmp[:, 7] = probs[:, 2] * probs[:, 4]
    tmp[:, 8] = probs[:, 2] * probs[:, 5]
    tmp[:, 9] = probs[:, 2] * probs[:, 6]
    tmp[:, 10] = probs[:, 3]
    return probs


def seg_loss(seg_pred, seg_true, seg_pos_weight=1, focal=0):
    if focal > 0:
        loss = nn.BCEWithLogitsLoss(reduction='none')(seg_pred, seg_true)
        alpha = seg_pos_weight
        probas = torch.sigmoid(seg_pred)
        loss = torch.where(seg_true >= 0.5, alpha * (1. - probas) ** focal * loss, probas ** focal * loss)
        return loss.mean()
    else:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(seg_pos_weight))(seg_pred, seg_true)


def cls_loss(cls_pred, cls_true, annotated=None, annotated_weight=1.0, focal=0, ce=False):
    if ce: return ce_loss(cls_pred, cls_true)
    loss = nn.BCEWithLogitsLoss(reduction='none')(cls_pred, cls_true)
    if focal > 0:
        alpha = 1.
        probas = torch.sigmoid(cls_pred)
        loss = torch.where(cls_true >= 0.5, alpha * (1. - probas) ** focal * loss, probas ** focal * loss)
    loss = loss.mean(dim=1)
    if annotated_weight != 1:
        loss = torch.where(annotated == 1, loss * annotated_weight, loss)
    return loss.mean()


def ce_loss(cls_pred, cls_true):
    logprobs = torch.nn.functional.log_softmax(cls_pred, dim=1)
    return -(cls_true * logprobs).sum() / cls_pred.shape[0]


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
    return score / pred.shape[1]


class Attn(nn.Module):
    def __init__(self, cfg, dim):
        super(Attn, self).__init__()
        tmp = cfg.resolution // 32
        seq_len = (tmp // 2) ** 2 if cfg.downconv else tmp ** 2
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
        pos_emb_channels = 1 if self.cfg.pos_emb else 0
        if cfg.downconv:
            self.avgpool = nn.AvgPool2d(2)
            self.downconv = nn.Sequential(
                nn.Conv2d(1, cfg.in_channels - 1, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(cfg.in_channels - 1),
                nn.ReLU(),
            )
        if cfg.init_random:
            encoder_weights = None
        else:
            if 'efficientnet' in cfg.encoder:
                encoder_weights = "noisy-student"
            elif 'resnext' in cfg.encoder:
                encoder_weights = 'swsl'
            else:
                encoder_weights = 'imagenet'

        target_size = 8 if cfg.two_view else 11
        seg_target_size = 1 if cfg.black_white else 11
        if cfg.pretrained_path is not None:
            if 'black_white' in cfg.pretrained_path:
                seg_target_size = 1
        self.unet = smp.Unet(
            encoder_name=cfg.encoder,
            # encoder_depth=cfg.encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=cfg.in_channels + pos_emb_channels * 2 if cfg.downconv else 1 + pos_emb_channels * 2,
            classes=seg_target_size,
            activation=None
        )
        self.encoder_out_channels = self.unet.encoder._out_channels[-1]
        self.unet.classification_head = nn.Sequential(
            nn.LayerNorm(self.encoder_out_channels),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.encoder_out_channels, target_size),
        )
        self.catheter_head = nn.Sequential(
            nn.LayerNorm(self.encoder_out_channels),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.encoder_out_channels, 4),
        )

        if self.cfg.use_attn:
            self.attn = Attn(cfg, self.encoder_out_channels)
        self.supcon_head = nn.Sequential(
            nn.Linear(self.encoder_out_channels, self.encoder_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_out_channels, 32 * target_size)
        )
        if self.cfg.last_conv:
            kernel_size = self.cfg.resolution // 32
            self.last_conv = nn.Sequential(
                # nn.BatchNorm2d(dim),
                nn.Dropout(cfg.dropout),
                nn.Conv2d(self.encoder_out_channels, self.encoder_out_channels,
                          kernel_size=kernel_size // 2 if self.cfg.downconv else kernel_size,
                          groups=self.encoder_out_channels // self.cfg.last_conv)
            )

    def forward(self, x, return_features=False):
        if self.cfg.downconv:
            x = torch.cat((self.avgpool(x), self.downconv(x)), dim=1)
        resolution = x.shape[-1]
        if self.cfg.pos_emb:
            pos_emb = torch.linspace(-1, 1, resolution)
            pos_emb_h = pos_emb.view(1, 1, -1, 1).expand(x.shape[0], 1, resolution, resolution).to(x.device)
            pos_emb_w = pos_emb.view(1, 1, 1, -1).expand(x.shape[0], 1, resolution, resolution).to(x.device)
            x = torch.cat((x, pos_emb_h, pos_emb_w), dim=1)
        if self.cfg.seg_pretrain or self.cfg.black_white:
            features = self.unet.encoder(x)
            seg_pred = self.unet.segmentation_head(self.unet.decoder(*features))
            cls_pred = torch.zeros(x.shape[0], 11).to(x.device)
            supcon_feats = None
        else:
            last_features = self.unet.encoder(x)[-1]
            if self.cfg.use_attn:
                last_features = self.attn(last_features)
            if self.cfg.last_conv:
                head_inputs = self.last_conv(last_features).squeeze(-1).squeeze(-1)
            else:
                head_inputs = last_features.mean(dim=(2, 3))
            if return_features:
                return head_inputs
            if self.cfg.supcon:
                cls_pred = torch.zeros(x.shape[0], 11).to(x.device)
                seg_pred = torch.zeros(x.shape[0], 11, resolution, resolution).to(x.device)
                supcon_feats = F.normalize(
                    self.supcon_head(self.unet.encoder(x)[-1].mean(dim=(2, 3))).reshape(x.shape[0], -1, 11), dim=1)
            else:
                if self.cfg.catheter:
                    cls_pred = self.catheter_head(head_inputs)
                else:
                    cls_pred = self.unet.classification_head(head_inputs)
                seg_pred = torch.zeros(x.shape[0], 11, resolution, resolution).to(x.device)
                supcon_feats = None
        return cls_pred, seg_pred, supcon_feats


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


class EncoderModel(nn.Module):
    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        self.cfg = cfg
        if cfg.downconv:
            self.avgpool = nn.AvgPool2d(2)
            self.downconv = nn.Sequential(
                nn.Conv2d(1, cfg.in_channels - 1, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(cfg.in_channels - 1),
                nn.ReLU(),
            )
        self.encoder = timm.create_model(cfg.encoder, pretrained=not bool(cfg.init_random), num_classes=0,
                                         global_pool='')
        patch_first_conv(self.encoder, cfg.in_channels + 2 if cfg.downconv else 1 + 2)
        dim = self.encoder.conv_head.out_channels
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim, 24)
        )

    def forward(self, x, return_features=False):
        if self.cfg.downconv:
            x = torch.cat((self.avgpool(x), self.downconv(x)), dim=1)
        resolution = x.shape[-1]
        if self.cfg.pos_emb:
            pos_emb = torch.linspace(-1, 1, resolution)
            pos_emb_h = pos_emb.view(1, 1, -1, 1).expand(x.shape[0], 1, resolution, resolution).to(x.device)
            pos_emb_w = pos_emb.view(1, 1, 1, -1).expand(x.shape[0], 1, resolution, resolution).to(x.device)
            x = torch.cat((x, pos_emb_h, pos_emb_w), dim=1)
        features = self.encoder(x).mean(dim=(2, 3))
        if return_features:
            return features
        logits = self.head(features)
        return logits


class SWAVWrapper(nn.Module):
    def __init__(self, model, cfg):
        super(SWAVWrapper, self).__init__()
        self.root = model
        self.cfg = cfg
        self.projection_head = nn.Sequential(
            nn.Linear(self.root.encoder_out_channels, cfg.hidden_mlp),
            nn.BatchNorm1d(cfg.hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_mlp, cfg.feat_dim),
        )
        if isinstance(cfg.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(cfg.feat_dim, cfg.nmb_prototypes)
        elif cfg.nmb_prototypes > 0:
            self.prototypes = nn.Linear(cfg.feat_dim, cfg.nmb_prototypes, bias=False)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.root(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True), return_features=True)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        output = self.projection_head(output)
        output = nn.functional.normalize(output, dim=1, p=2)
        return output, self.prototypes(output)


class SIMSIAMWrapper(nn.Module):
    def __init__(self, model, cfg):
        super(SIMSIAMWrapper, self).__init__()
        self.root = model
        self.cfg = cfg
        hidden_dim = 2048
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.root.encoder_out_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False)
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

    def asymmetric_loss(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def simsiam_loss(self, z1, z2, p1, p2):
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)
        return 0.5 * loss1 + 0.5 * loss2

    def forward(self, img1, img2):
        features1 = self.root(img1, return_features=True)
        z1 = self.projection_mlp(features1)
        p1 = self.prediction_mlp(z1)
        features2 = self.root(img2, return_features=True)
        z2 = self.projection_mlp(features2)
        p2 = self.prediction_mlp(z2)
        return self.simsiam_loss(z1, z2, p1, p2)
