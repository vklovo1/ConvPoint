import sys

sys.path.append('../../')

import numpy as np
import argparse
import os
from tqdm import tqdm

import torch
import torch.utils.data

import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from networks.network_seg import SegBig as Net

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

N_CLASSES = 8
MODEL_PATH = "./models_SEMANTIC3D_v0/SegBig_rgb/state_dict.pth"


class PartDatasetTest:

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:, 0] < pt[0] + bs / 2, self.xyzrgb[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzrgb[:, 1] < pt[1] + bs / 2, self.xyzrgb[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder,
                 block_size=8,
                 npoints=8192,
                 test_step=0.8, nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = nocolor
        self.filename = filename

        # load the points
        self.xyzrgb = np.load(os.path.join(self.folder, self.filename), fix_imports=True,
                              encoding='latin1')
        step = test_step
        discretized = ((self.xyzrgb[:, :2]).astype(float) / step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float) * step

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points
        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:, 3:6]
            fts = fts.astype(np.float32)
            fts = fts / 255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return len(self.pts)


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    print(indices.shape)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', '-s', help='Path to data folder')
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument('--block_size', help='Block size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--iter", "-i", type=int, default=1000)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", default=0.8, type=float)
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--drop", default=0.5, type=float)
    args = parser.parse_args()

    model = Net(3, 8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(model)

    for filename in filelist_test:
        ds = PartDatasetTest(filename, args.rootdir,
                             block_size=args.block_size,
                             npoints=args.npoints,
                             test_step=args.test_step,
                             nocolor=args.nocolor
                             )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.threads
                                             )

        xyzrgb = ds.xyzrgb[:, :3]
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for pts, features, indices in t:
                outputs = model(features, pts)

                outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                scores[indices.cpu().numpy().ravel()] += outputs_np

        mask = np.logical_not(scores.sum(1) == 0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

        # compute softmax
        scores = scores - scores.max(axis=1)[:, None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
        scores = np.nan_to_num(scores)

        os.makedirs(os.path.join(args.savedir, "results"), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, "results", filename)
        scores = scores.argmax(1)
        np.savetxt(save_fname, scores, fmt='%d')

        if args.savepts:
            save_fname = os.path.join(args.savedir, "results", f"{filename}_pts.txt")
            xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores, 1)], axis=1)
            np.savetxt(save_fname, xyzrgb, fmt=['%.4f', '%.4f', '%.4f', '%d'])


if __name__ == '__main__':
    main()
