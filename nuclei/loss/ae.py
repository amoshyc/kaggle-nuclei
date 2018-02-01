import numpy as np
import torch
from torch.autograd import Variable

from nuclei import config
from nuclei.features.raw import *


def tag_loss_single(tag_map, mask_coords):
    n_masks = len(mask_coords)
    re = Variable(torch.zeros((n_masks)).cuda(), requires_grad=False)
    loss1 = Variable(torch.zeros((n_masks)).cuda(), requires_grad=False)
    for i, (rr, cc) in enumerate(mask_coords):
        re[i] = torch.mean(tag_map[rr, cc])
    for i, (rr, cc) in enumerate(mask_coords):
        loss1[i] = torch.mean((tag_map[rr, cc] - re[i])**2)
    A = re.expand(n_masks, n_masks)
    B = torch.transpose(A, 0, 1)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(torch.exp(-1/2 * (A - B) ** 2))
    return loss1 + loss2


def tag_loss(tag_map_batch, mask_coords_batch, pbar=False):
    assert len(tag_map_batch) == len(mask_coords_batch)
    if pbar:
        tag_map_batch = tqdm(tag_map_batch)

    loss = 0.0
    for tag_map, mask_coords in zip(tag_map_batch, mask_coords_batch):
        loss += tag_loss_single(tag_map, mask_coords)
    return loss / len(tag_map_batch)


def tag_loss_single_np(tag_map, mask_coords):
    n_masks = len(mask_coords)
    re = np.zeros((n_masks), dtype=np.float32)
    loss1 = np.zeros((n_masks), dtype=np.float32)
    for i, (rr, cc) in enumerate(mask_coords):
        re[i] = np.mean(tag_map[rr, cc])
    for i, (rr, cc) in enumerate(mask_coords):
        loss1[i] = np.mean((tag_map[rr, cc] - re[i])**2)
    A = np.broadcast_to(re, (n_masks, n_masks))
    B = A.T
    loss2 = np.exp(-1 / 2 * (A - B) ** 2)
    loss1 = np.mean(loss1)
    loss2 = np.mean(loss2)
    return loss1 + loss2


