import torch
import torch.nn as nn
import torch.nn.functional as F

bce = nn.BCELoss(reduction='mean')
bce_with_logits = nn.BCEWithLogitsLoss(reduction='mean')
mse = nn.MSELoss(reduction='mean')


def multi_bce(preds, gt):
    m_loss = bce(preds[2], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        loss += bce(preds[i], gt) / 4
    return loss + m_loss, m_loss


def multi_bce_2outputs(preds, gt):
    m_loss = bce_with_logits(preds[1], gt)
    loss = bce_with_logits(preds[0], gt) / 2
    return loss + m_loss, m_loss


def multi_bce_2(preds, gt):
    m_loss = bce(preds[2], gt)
    loss = 0.
    loss += bce(preds[0], gt) / 4
    loss += structure_loss(preds[1], gt) / 4
    return loss + m_loss, m_loss


def mse_loss(pred, gt):
    """
    计算均方误差损失
    Args:
        pred: 预测结果张量
        gt: 真实标签张量
    Returns:
        损失值
    """
    if isinstance(pred, dict):
        pred = pred['sal']
    return mse(pred, gt)


def single_bce(pred, gt):
    return bce(pred, gt)


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    pred is logits
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') 
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)  # if pred is logits
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def mse_structure_loss(pred, mask):
    """
    mse_structure_loss 的 Docstring
    :param preds: 说明
    :param mask: 说明
    :return total, mes
    """
    pred_sigmoid = torch.sigmoid(pred)
    main_mse = mse(pred_sigmoid, mask)
    main_structure = structure_loss(pred, mask)
    # total_loss = 0.6 * main_mse + 0.4 * main_structure
    return main_structure, main_mse
