import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import logging
from option import args

def get_logger(log_dir, args):
    ''' LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_dir(args):
    experiment_dir = Path(args.path_log)
    experiment_dir.mkdir(exist_ok=True)
    task_path = 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + str(args.scale_factor) + 'x'

    experiment_dir = experiment_dir.joinpath(task_path)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model_name)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.data_name)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    return experiment_dir, checkpoints_dir, log_dir


class Logger(object):
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, string):
        if args.local_rank <= 0:
            self.logger.info(string)
            print(string)




def cal_metrics(args, label, out):
    if len(label.size()) == 2:
        label = rearrange(label, '(a1 h) (a2 w) -> 1 1 a1 h a2 w', a1=args.angRes, a2=args.angRes)
        out = rearrange(out, '(a1 h) (a2 w) -> 1 1 a1 h a2 w', a1=args.angRes, a2=args.angRes)
        
    if len(label.size()) == 4:
        [B, C, H, W] = label.size()
        label = label.view((B, C, args.angRes, H//args.angRes, args.angRes, H//args.angRes))
        out = out.view((B, C, args.angRes, H // args.angRes, args.angRes, W // args.angRes))

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(),
                                                                out_y[b, u, :, v, :].numpy())
                SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                              out_y[b, u, :, v, :].numpy(),
                                                              gaussian_weights=True, )

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE:u*hE+h, v*wE:v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = \
                        dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    # H, W = numU*pH, numV*pW
    ph, pw = pH // angRes, pW // angRes
    bdr = (pz - stride) // 2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = \
                        subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]

            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    # y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255
    y[:, :, 0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:, :, 1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:, :, 2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat) * 255

    y = np.zeros(x.shape, dtype='double')
    y[:, :, 0] =  mat_inv[0, 0] * x[:, :, 0] + mat_inv[0, 1] * x[:, :, 1] + mat_inv[0, 2] * x[:, :, 2] -  16.0 / 255.0
    y[:, :, 1] =  mat_inv[1, 0] * x[:, :, 0] + mat_inv[1, 1] * x[:, :, 1] + mat_inv[1, 2] * x[:, :, 2] - 128.0 / 255.0
    y[:, :, 2] =  mat_inv[2, 0] * x[:, :, 0] + mat_inv[2, 1] * x[:, :, 1] + mat_inv[2, 2] * x[:, :, 2] - 128.0 / 255.0

    return y


def crop_center_view(data, angRes_in, angRes_out):
    assert angRes_in >= angRes_out, 'angRes_in requires to be greater than angRes_out'
    [B, _, H, W] = data.size()
    patch_size = H // angRes_in
    data = data[:, :,
           (angRes_in - angRes_out) // 2 * patch_size:(angRes_in + angRes_out) // 2 * patch_size,
           (angRes_in - angRes_out) // 2 * patch_size:(angRes_in + angRes_out) // 2 * patch_size]

    return data


def cal_loss_class(probability):
    assert len(probability.size()) == 2, 'probability requires a 2-dim tensor'
    [B, num_cluster] = probability.size()
    loss_class = 0
    for batch in range(B):
        sum_re = 0
        for i in range(num_cluster - 1):
            for j in range(i + 1, num_cluster):
                sum_re += abs(probability[batch][i] - probability[batch][j])

        loss_class += ((num_cluster - 1) - sum_re)
    loss_class = loss_class / B

    return loss_class
