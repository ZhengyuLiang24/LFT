from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict


def main(args):
    ''' Create Dir for Save'''
    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda '''
    torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    device = torch.device("cpu", args.local_rank)

    ''' DATA TEST LOADING '''
    logger.log_string('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of test data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' load pre-trained pth '''
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        logger.log_string('Use pretrain model!')
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        logger.log_string('Use pretrain model!')

    net = net.to(device)
    cudnn.benchmark = True

    ''' TEST on every dataset'''
    logger.log_string('\nStart test...')
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            psnr_epoch_test, ssim_epoch_test = test(test_loader, device, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                start_epoch, test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass
    pass


def test(test_loader, device, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Hr_SAI_y = Hr_SAI_y.squeeze()

        uh, vw = Lr_SAI_y.shape
        h0, w0 = int(uh//args.angRes), int(vw//args.angRes)

        subLFin = LFdivide(Lr_SAI_y, args.angRes, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFout = torch.zeros(numU, numV, args.angRes * args.patch_size_for_test * args.scale_factor,
                               args.angRes * args.patch_size_for_test * args.scale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u:u+1, v:v+1, :, :]
                with torch.no_grad():
                    net.eval()
                    torch.cuda.empty_cache()
                    out = net(tmp.to(device))
                    subLFout[u:u+1, v:v+1, :, :] = out.squeeze()

        Sr_4D_y = LFintegrate(subLFout, args.angRes, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, h0 * args.scale_factor,
                              w0 * args.scale_factor)
        Sr_SAI_y = Sr_4D_y.permute(0, 2, 1, 3).reshape((h0 * args.angRes * args.scale_factor,
                                                        w0 * args.angRes * args.scale_factor))

        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    from option import args

    main(args)
