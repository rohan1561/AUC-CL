import torch
import math


def sim_matrix_calc(a, b, mode, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    # range mapping to 0, 1
    slope = 1/2
    sim_mt = slope*(sim_mt + 1)
    return sim_mt


def sim_matrix_calc2(a, b, mode):
    """
    added eps for numerical stability
    """
    eps=1e-8
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    # range mapping to 0, 1
    slope = 1/2
    sim_mt = slope*(sim_mt + 1)

    if mode == 'train':
        mask_pos = torch.eye(a.shape[0], device=a.device)
        sim_mt = sim_mt - 0.5*mask_pos
        sim_mt = sim_mt.clamp(0, 1)
    return sim_mt


def sim_matrix_arc(a, b, mode, m=0.5, easy_margin=False):
    eps=1e-8
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    cosine = torch.mm(a_norm, b_norm.transpose(0, 1))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    phi = cosine * cos_m - sine * sin_m

    if easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > th, phi, cosine - mm)

    if mode == 'train':
        mask_pos = torch.eye(a.shape[0], device=a.device)
        mask_neg = (torch.ones_like(cosine) - torch.eye(a.shape[0],\
            device=a.device))
        output = mask_pos * phi + mask_neg * cosine
    else:
        output = cosine

    return output

