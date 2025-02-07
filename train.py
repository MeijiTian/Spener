import os
import SimpleITK as sitk
import numpy as np
import torch
import dataset
import tinycudann as tcnn
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import *
from skimage.transform import radon, iradon
import argparse
import netarch
from bm3d import bm3d
import torch.nn.functional as F

def eval_model(model, val_loader, test_loader, prior_img, device):
    model.eval()
    img_size = prior_img.shape[0]
    D = int(img_size * (2**0.5))
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    recon = None

    with torch.no_grad():
        for i, (ray_sample) in enumerate(val_loader):
            ray_sample = ray_sample.to(device).float()
            pre_intensity = model(prior_tensor, ray_sample).view(-1, 2 * D, 2 * D, 1)
            _pre_img = pre_intensity[0].squeeze().float().cpu().numpy()
            _pre_img = _pre_img[(2 * D - img_size) // 2: (2 * D + img_size) // 2,
                             (2 * D - img_size) // 2: (2 * D + img_size) // 2]
        recon = _pre_img

    return recon

def train_w_sino(model, prior_img, lr, train_epoch, train_loader, val_loader, test_loader, device, dv_fbp, result_path):

    img_size = prior_img.shape[0]
    D = int(img_size * (2**0.5))
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loop = tqdm(range(train_epoch), colour = 'green', ncols=120)
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    summary_epoch = train_epoch // 5

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for e in train_loop:
        model.train()
        loss_train = 0
        for _, (ray, proj_l) in enumerate(train_loader):
            batch_size, sample_N = ray.shape[0], ray.shape[1]
            ray, proj_l = ray.to(device).float(), proj_l.to(device).float()
            pre_intensity = model(prior_tensor, ray)
            pre_intensity = pre_intensity.reshape(-1, sample_N, 2*D, 1)
            pre_proj_l = torch.sum(pre_intensity, dim=2).squeeze(-1).float()

            loss = loss_fun(pre_proj_l, proj_l.to(pre_proj_l.dtype))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        train_loop.set_description(f'[{e+1}/{train_epoch}], lr:{optimizer.param_groups[0]["lr"]:.6f}'
                                f' Loss:{loss_train/len(train_loader):.6f}')

        if (e+1) % summary_epoch == 0 or e == train_epoch - 1:

            recon = eval_model(model, val_loader, test_loader, prior_img, device)

            psnr, ssim = cal_psnr(recon, dv_fbp), cal_ssim(recon, dv_fbp)
            print(f'Epoch {e+1},  PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')
            sitk.WriteImage(sitk.GetImageFromArray(recon), os.path.join(result_path, f'recon_{e+1}.nii.gz'))
            torch.save(model.state_dict(), os.path.join(result_path, f'model_{e+1}.pth'))

    return model, recon

def train_w_reg(model, prior_img, reg_img, lr, train_epoch, train_loader, val_loader, test_loader, device, dv_fbp, result_path, lamda):

    img_size = prior_img.shape[0]
    D = int(img_size * (2**0.5))
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loop = tqdm(range(train_epoch), colour = 'green', ncols=120)
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    reg_tensor = generate_prior_image_for_fanbeam(prior_img=reg_img, D = D, device=device)

    summary_epoch = train_epoch // 5

    train_coord = dataset.build_coordinate_val(D, 0)
    train_coord = torch.tensor(train_coord, dtype=torch.float32).to(device)
    patch_size = 128

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for e in train_loop:
        model.train()
        loss_sino = 0
        loss_reg = 0
        for _, (ray, proj_l) in enumerate(train_loader):
            batch_size, sample_N = ray.shape[0], ray.shape[1]
            ray, proj_l = ray.to(device).float(), proj_l.to(device).float()
            pre_intensity = model(prior_tensor, ray)
            pre_intensity = pre_intensity.reshape(-1, sample_N, 2*D, 1)
            pre_proj_l = torch.sum(pre_intensity, dim=2).squeeze(-1).float()

            sino_loss = loss_fun(pre_proj_l, proj_l.to(pre_proj_l.dtype))
            start_x, start_y = np.random.randint(200, 2*D - 200 - patch_size), np.random.randint(200, 2 * D - 200 - patch_size)
            select_coord = train_coord[:, start_x:start_x + patch_size, start_y:start_y + patch_size]
            select_intensity = reg_tensor[:, :, start_x:start_x + patch_size, start_y:start_y + patch_size].flatten()
            pre_intensity = model(prior_tensor, select_coord)
            pre_intensity = pre_intensity.flatten()
            reg_loss = loss_fun(pre_intensity, select_intensity)
            loss = sino_loss + lamda * reg_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sino += sino_loss.item()
            loss_reg += reg_loss.item()

        train_loop.set_description(f'[{e+1}/{train_epoch}], lr:{optimizer.param_groups[0]["lr"]:.6f}'
                                f' Sino Loss:{loss_sino/len(train_loader):.6f} Reg Loss:{loss_reg/len(train_loader):.6f}')

        if (e + 1) % summary_epoch == 0 or e == train_epoch - 1:
            recon = eval_model(model, val_loader, test_loader, prior_img, device)

            psnr, ssim = cal_psnr(recon, dv_fbp), cal_ssim(recon, dv_fbp)
            print(f'Epoch {e + 1},  PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')
            sitk.WriteImage(sitk.GetImageFromArray(recon),
                            os.path.join(result_path, f'recon_{e + 1}.nii.gz'))

            torch.save(model.state_dict(), os.path.join(result_path, f'model_{e + 1}.pth'))

    return model, recon

def train(config):

    ## load_config_content
    input_path = config["file"]["input_path"]
    result_path = config["file"]["result_path"]
    model_path = config["file"]["model_path"]
    num_sv, num_dv, img_size = config["file"]["num_sv"], config["file"]["num_dv"], config["file"]["img_size"]
    I0 = config["file"]["I0"]

    lr = config["train"]["lr"]
    sample_N = config["train"]["sample_N"]

    # training parameters
    gpu = config['train']['gpu']
    lamda = config['train']['lamda']
    iter_num = config['train']['iter_num']

    spacing = config['file']['fan_geometry']['spacing']
    det_count = config['file']['fan_geometry']['det_count']
    D = config['file']['fan_geometry']['SOD']
    img_size = config['file']['img_size']

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
    if img_size != gt_img.shape[0]:
        raise ValueError('GT Image size does not match the input image size')

    sensor_pos = np.linspace(-det_count / 2, det_count / 2, det_count) * spacing
    fan_pose = np.rad2deg(np.arctan(sensor_pos / (2 * D)))
    L = len(fan_pose)

    projection_angle_num = num_dv
    scale = int(num_dv / num_sv)


    # generate dense-view sinogram data via fanbeam geometry
    dv_sino = fanbeam_projection(gt_img, D, det_count, spacing, projection_angle_num, device)
    sv_sino = dv_sino[::scale, :]
    

    if I0 is not None:
        taskname = f'SV_{num_sv}_I0_{I0:.0e}/'
        sv_sino = add_noise_to_sino(sv_sino, I0, 0.01)
    else:
        taskname = f'SV_{num_sv}/'

    result_path = os.path.join(result_path, taskname)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    sv_fbp_recon = fanbeam_backprojection(sv_sino, D, det_count, spacing, num_sv, device)
    
    # save img and print current taskname
    print('Taskname: ', result_path,'D:', D, 'L:', L, 'scale', scale)

    # compute psnr ssim with GT and DV_FBP
    psnr, ssim = cal_psnr(sv_fbp_recon, gt_img), cal_ssim(sv_fbp_recon, gt_img)
    print(f'SV_FBP Compared with GT Image PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')





    
    sitk.WriteImage(sitk.GetImageFromArray(sv_fbp_recon), os.path.join(result_path, 'sv_fbp.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(gt_img), os.path.join(result_path, 'gt_img.nii.gz'))

    # build dataset
    train_loader = data.DataLoader(
        dataset.TrainData(sin_data = sv_sino, theta=num_sv, sample_N=sample_N,
                        fan_pose= fan_pose, D=D),
        batch_size=config['train']['batch_size'],
        shuffle=True)

    val_loader = data.DataLoader(
        dataset=dataset.ValData(D=D, angle=0),
        batch_size=1,
        shuffle=False
    )
    test_loader = data.DataLoader(
        dataset=dataset.TestData(theta=projection_angle_num, D=D, fan_pose=fan_pose),
        batch_size=6,
        shuffle=False
    )

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # build model
    En_SCOPE = netarch.SpenerNet(encoder_config=config['encoding'], network_config=config['network']).to(device)
    
    sv_fbp_recon = np.clip(sv_fbp_recon, 0, 1)
    model, recon = train_w_sino(model=En_SCOPE, prior_img = sv_fbp_recon, lr=lr, train_epoch=1000,
                                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                        device=device, dv_fbp=gt_img, result_path=result_path + '1_iter/')
    lamda = lamda
    for iternum in range(iter_num):
        # adopting BM3D for removing noise
        recon = bm3d(recon, sigma_psd=0.01)
        sitk.WriteImage(sitk.GetImageFromArray(recon), os.path.join(result_path, f'denoised_recon_{iternum + 1}.nii.gz'))
        psnr, ssim = cal_psnr(recon, gt_img), cal_ssim(recon, gt_img)
        print(f'Iter {iternum + 1}, denoised  recon PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')

        model, recon = train_w_reg(model=model, prior_img=recon, reg_img=recon, lr=lr/10, train_epoch=250,
                                    train_loader=train_loader, val_loader=val_loader, test_loader= test_loader,
                                    device=device, dv_fbp=gt_img, result_path=result_path + f'{iternum + 2}_iter/', lamda = lamda)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spener')

    parser.add_argument('--config', type=str, default='config/demo_config.json', help='config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config)
    
