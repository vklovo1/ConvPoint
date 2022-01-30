import argparse
import os

from utils import data_utils
import numpy as np


label_colors = {
    0: [0, 0, 255],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [4, 61, 4],
    4: [90, 90, 90],
    5: [222, 33, 166],
    6: [255, 255, 255],
    7: [255, 255, 0]
}

filelist_test = [
    "birdfountain_station1_xyz_intensity_rgb_voxels.npy",
    "castleblatten_station1_intensity_rgb_voxels.npy",
    "castleblatten_station5_xyz_intensity_rgb_voxels.npy",
    "marketplacefeldkirch_station1_intensity_rgb_voxels.npy",
    "marketplacefeldkirch_station4_intensity_rgb_voxels.npy",
    "marketplacefeldkirch_station7_intensity_rgb_voxels.npy",
    "sg27_station10_intensity_rgb_voxels.npy",
    "sg27_station3_intensity_rgb_voxels.npy",
    "sg27_station6_intensity_rgb_voxels.npy",
    "sg27_station8_intensity_rgb_voxels.npy",
    "sg28_station2_intensity_rgb_voxels.npy",
    "sg28_station5_xyz_intensity_rgb_voxels.npy",
    "stgallencathedral_station1_intensity_rgb_voxels.npy",
    "stgallencathedral_station3_intensity_rgb_voxels.npy",
    "stgallencathedral_station6_intensity_rgb_voxels.npy",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pointsdir', '-s', help='Path to points folder')
    parser.add_argument('--masksdir', type=str, help='Path to masks folder', default="./results/")
    parser.add_argument("--savedir", type=str, default="./results")
    args = parser.parse_args()

    for filename in filelist_test:
        points_load_folder = os.path.join(args.pointsdir, f"{filename}")
        masks_load_folder = os.path.join(args.masksdir, f"{filename}")

        points = np.load(points_load_folder)
        masks = np.loadtxt(masks_load_folder)

        points_xyz = points[:, 0:3]
        points_colors = points[:, 3:6]

        masks_xyz = np.copy(points_xyz)
        mask_colors = np.array([label_colors[m] for m in masks])

        ply_save_folder = os.path.join(args.savedir, "ply")

        data_utils.save_ply(points_xyz, ply_save_folder + f"/{filename}_points_ply.ply", colors=points_colors)
        data_utils.save_ply(masks_xyz, ply_save_folder + f"/{filename}_mask_ply.ply", colors=mask_colors)


if __name__ == '__main__':
    main()
