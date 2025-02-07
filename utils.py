from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import torch
import torch_radon

def cal_psnr(recon, target):
    datarange = target.max() - target.min()
    return PSNR(target, recon, data_range=datarange)

def cal_ssim(recon, target):
    return SSIM(target, recon, data_range=target.max() - target.min())

def generate_prior_image_for_fanbeam(prior_img, D, device):

    img_size = prior_img.shape[0]
    prior_img_pad = np.zeros((2 * D, 2 * D))
    prior_img_pad[(2 * D - img_size) // 2 : (2 * D + img_size) // 2,
                  (2 * D - img_size) // 2: (2 * D + img_size) // 2] = prior_img
    img_tensor = torch.FloatTensor(prior_img_pad).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor

def add_noise_to_sino(sino, N0, k):
    sino_ = N0 * np.exp(-sino * k)
    noisy_sino = np.random.poisson(sino_)
    out_sino = -np.log(noisy_sino / N0) / k
    return out_sino

def fan_coordinate(fan_angle, D):
    origin_x = 1
    origin_y = 0
    x = np.linspace(-1, 1, int(2*D)).reshape(-1, 1)  # (2D, ) -> (2D, 1)
    y = np.zeros_like(x)  # (2D, 1)
    xy_temp = np.concatenate((x, y), axis=-1)  # (2D, 2)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)  # (2D, 3)

    L = len(fan_angle)
    xy = np.zeros(shape=(L, int(2*D), 2)) # (L, 2D, 2)
    for i in range(L):
        fan_angle_rad = np.deg2rad(fan_angle[L-i-1])
        M = np.array(
            [
                [np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
                -1*origin_x*np.cos(fan_angle_rad)+origin_y*np.sin(fan_angle_rad)+origin_x],
                [np.sin(fan_angle_rad), np.cos(fan_angle_rad),
                -1*origin_x*np.sin(fan_angle_rad)-origin_y*np.cos(fan_angle_rad)+origin_y],
                [0, 0, 1]
            ]
        )
        temp = xy_temp @ M.T # (2D, 3) @ (3, 3) -> (2D, 3)
        xy[i, :, :] = temp[:, :2]
    return xy


def build_coordinate_train(xy, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ]
    )
    L, D_2, _ = xy.shape
    xy = xy.reshape(-1, 2)  # (L*2D, 2)
    xy = xy @ trans_matrix.T  # (L*2D, 2) @ (2, 2) -> (L*2D, 2)
    xy = xy.reshape(L, D_2, 2)  # (L, 2D, 2)
    return xy


def fanbeam_projection(img, D, det_count, spacing, num_angles, device):

    fanbeam_angles = np.linspace(0, 360, num_angles, endpoint=False)
    fanbeam_angles = np.deg2rad(fanbeam_angles)
    randon_fanbeam = torch_radon.FanBeam(det_count = det_count,
                                        angles = fanbeam_angles,
                                        src_dist = D,
                                        det_dist = D,
                                        det_spacing = spacing)
    x = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    sinogram = randon_fanbeam.forward(x)
    sino_data = sinogram.squeeze().cpu().numpy()
    return sino_data



def fanbeam_backprojection(sino, D, det_count, spacing, num_angles, device):

    with torch.no_grad():
        fanbeam_angles = np.linspace(0, 360, num_angles, endpoint=False)
        fanbeam_angles = np.deg2rad(fanbeam_angles)
        randon_fanbeam = torch_radon.FanBeam(det_count = det_count,
                                            angles = fanbeam_angles,
                                            src_dist = D,
                                            det_dist = D,
                                            det_spacing = spacing)
        sino_tensor = torch.tensor(sino, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_tensor = torch.zeros((1, 1, 256, 256), dtype=torch.float32).to(device)
        sino_test = randon_fanbeam.forward(x_tensor)
        filter_sino = randon_fanbeam.filter_sinogram(sino_tensor)
        dv_fbp_recon = randon_fanbeam.backward(filter_sino)
        dv_fbp_recon = dv_fbp_recon.squeeze().cpu().numpy()
    return dv_fbp_recon




