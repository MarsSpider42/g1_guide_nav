import argparse
import numpy as np
import open3d as o3d
from lidar_process.old_files.lidar_scan_storage import ScanStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted_dir", help="folder produced by init_bag_storage.py for one bag")
    parser.add_argument("index", type=int, help="scan index to view")
    parser.add_argument("--color-by", choices=["height", "reflectivity", "none"], default="height",
                         help="color points by z-height, reflectivity, or plain (default: height)")
    args = parser.parse_args()

    store = ScanStore(args.extracted_dir)
    print(f"{len(store)} scans total, format={store.meta.get('format')}, topic={store.meta.get('topic')}")

    scan = store[args.index]
    print(f"scan {args.index}: t={store.timestamp(args.index)}, {store.point_num(args.index)} points")

    xyz = np.stack([scan["x"], scan["y"], scan["z"]], axis=-1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if args.color_by == "height":
        z = scan["z"].astype(np.float64)
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        colors = np.stack([z_norm, np.zeros_like(z_norm), 1 - z_norm], axis=-1)  # blue (low) -> red (high)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif args.color_by == "reflectivity":
        r = scan["reflectivity"].astype(np.float64) / 255.0
        colors = np.stack([r, r, r], axis=-1)  # grayscale
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # "none" leaves points uncolored (Open3D default)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Scan {args.index} (t={store.timestamp(args.index)})",
        width=1024,
        height=768,
    )