import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data

def build_coordinate_val(D, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ]
    )

    x = np.linspace(-1, 1, int(2 * D)).reshape(-1, 1)  # (2D, ) -> (2D, 1)
    y = np.linspace(-1, 1, int(2 * D)).reshape(-1, 1)  # (2D, ) -> (2D, 1)
    x, y = np.meshgrid(x, y, indexing='ij')  # (2D, 2D)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # ((2D * 2D), 2)
    xy = xy @ trans_matrix.T  # (2D * 2D, 2) @ (2, 2)
    xy = xy.reshape(1, 2 * D, 2 * D, 2)
    return xy


class TestData(data.Dataset):
    def __init__(self, theta, D, fan_pose):
        # rotation angles
        angles = np.linspace(0, 360, theta+1)
        angles = angles[:len(angles) - 1]
        num_angles = len(angles)
        # fan angles
        fan_angle = fan_pose # (1, L) -> (L,)
        self.rays = []
        for i in range(num_angles):
            xy = utils.fan_coordinate(fan_angle, D) # (L, 2D, 2)
            self.rays.append(utils.build_coordinate_train(xy, angles[i])) # (num_angles, L, 2D, 2)

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, 2D, 2)
        return ray

class ValData(data.Dataset):
    def __init__(self, D, angle):

        self.rays = build_coordinate_val(D, angle)

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]
        return ray



class TrainData(data.Dataset):
    def __init__(self, theta, sin_data, sample_N, fan_pose, D):
        self.sample_N = sample_N
        # rotation angles
        angles = np.linspace(0, 360, theta+1)
        angles = angles[:len(angles) - 1]
        num_angles = len(angles)
        # fan angles
        fan_angle = fan_pose # (1, L) -> (L,)
        # sinogram
        sin = sin_data
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            xy = utils.fan_coordinate(fan_angle, D) # (L, 2D, 2)
            self.rays.append(utils.build_coordinate_train(xy, angles[i])) # (num_angles, L, 2D, 2)

        self.projections_lines = np.array(self.projections_lines)
        self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        projection_l = self.projections_lines[item]     # (L, )
        ray = self.rays[item]   # (L, 2D, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]    # (sample_N, 2D, 2)
        return ray_sample, projection_l_sample

