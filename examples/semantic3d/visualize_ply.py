import open3d
import argparse
import os

filelist_test = [
    "birdfountain_station1_xyz_intensity_rgb_voxels.npy_points_ply.ply",
    "castleblatten_station1_intensity_rgb_voxels.npy_points_ply.ply",
    "castleblatten_station5_xyz_intensity_rgb_voxels.npy_points_ply.ply",
    "marketplacefeldkirch_station1_intensity_rgb_voxels.npy_points_ply.ply",
    "marketplacefeldkirch_station4_intensity_rgb_voxels.npy_points_ply.ply",
    "marketplacefeldkirch_station7_intensity_rgb_voxels.npy_points_ply.ply",
    "sg27_station10_intensity_rgb_voxels.npy_points_ply.ply",
    "sg27_station3_intensity_rgb_voxels.npy_points_ply.ply",
    "sg27_station6_intensity_rgb_voxels.npy_points_ply.ply",
    "sg27_station8_intensity_rgb_voxels.npy_points_ply.ply",
    "sg28_station2_intensity_rgb_voxels.npy_points_ply.ply",
    "sg28_station5_xyz_intensity_rgb_voxels.npy_points_ply.ply",
    "stgallencathedral_station1_intensity_rgb_voxels.npy_points_ply.ply",
    "stgallencathedral_station3_intensity_rgb_voxels.npy_points_ply.ply",
    "stgallencathedral_station6_intensity_rgb_voxels.npy_points_ply.ply",
    "birdfountain_station1_xyz_intensity_rgb_voxels.npy_mask_ply.ply",
    "castleblatten_station1_intensity_rgb_voxels.npy_mask_ply.ply",
    "castleblatten_station5_xyz_intensity_rgb_voxels.npy_mask_ply.ply",
    "marketplacefeldkirch_station1_intensity_rgb_voxels.npy_mask_ply.ply",
    "marketplacefeldkirch_station4_intensity_rgb_voxels.npy_mask_ply.ply",
    "marketplacefeldkirch_station7_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg27_station10_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg27_station3_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg27_station6_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg27_station8_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg28_station2_intensity_rgb_voxels.npy_mask_ply.ply",
    "sg28_station5_xyz_intensity_rgb_voxels.npy_mask_ply.ply",
    "stgallencathedral_station1_intensity_rgb_voxels.npy_mask_ply.ply",
    "stgallencathedral_station3_intensity_rgb_voxels.npy_mask_ply.ply",
    "stgallencathedral_station6_intensity_rgb_voxels.npy_mask_ply.ply",
]


def set_up_camera(file_path):
    cloud = open3d.io.read_point_cloud(file_path)
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.run()


def main(args):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    params = open3d.io.read_pinhole_camera_parameters(os.path.join(args.savedir, "CameraParams.json"))
    ctr.convert_from_pinhole_camera_parameters(params)

    for filename in filelist_test:
        ply_load_folder = os.path.join(args.plydir, f"{filename}")
        img_save_folder = os.path.join(args.savedir, f"{filename}.jpg")
        cloud = open3d.io.read_point_cloud(ply_load_folder)
        vis.add_geometry(cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(img_save_folder)
        vis.remove_geometry(cloud)

    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plydir', '-s', help='Path to points folder', default="./results/ply")
    parser.add_argument("--savedir", type=str, default="./results/img")
    parser.add_argument("--setupmode", action="store_true")
    argum = parser.parse_args()

    if argum.setupmode:
        set_up_camera(os.path.join(argum.plydir, f"{filelist_test[0]}"))
    else:
        main(argum)
