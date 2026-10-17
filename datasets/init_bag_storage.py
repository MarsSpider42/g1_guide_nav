import argparse
from pathlib import Path
from lidar_scan_storage import summarize_bag, extract_scans


def init_all_bags(bags_root, output_root):
    bags_root, output_root = Path(bags_root), Path(output_root)
    bag_dirs = sorted(p for p in bags_root.iterdir() if p.is_dir())

    for bag_dir in bag_dirs:
        out_dir = output_root / bag_dir.name
        if (out_dir / "index.npy").exists():
            print(f"Skipping {bag_dir.name} (already extracted)")
            continue
        n, topic, fmt = extract_scans(bag_dir, out_dir)
        print(f"[{bag_dir.name}] {n} scans ({fmt}, topic={topic}) -> {out_dir/'scans'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to a single bag, or a folder of bags with --all")
    parser.add_argument("output", help="output folder for extracted scans")
    parser.add_argument("--all", action="store_true", help="treat input as a folder containing multiple bags")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    parser.add_argument("--summary-only", action="store_true", help="just print scan count/rate, don't extract")
    args = parser.parse_args()

    if args.summary_only:
        if args.all:
            for bag_dir in sorted(Path(args.input).iterdir()):
                if bag_dir.is_dir():
                    print(bag_dir.name, summarize_bag(bag_dir))
        else:
            print(summarize_bag(args.input))
    elif args.all:
        init_all_bags(args.input, args.output)
    else:
        n, topic, fmt = extract_scans(args.input, args.output)
        print(f"{n} scans ({fmt}, topic={topic}) -> {Path(args.output)/'scans'}")


'''python init_scans.py ~/bags/session_2026-09-14 ~/bags/extracted/session_2026-09-14
python init_scans.py ~/bags/raw ~/bags/extracted --all
python init_scans.py ~/bags/session_2026-09-14 - --summary-only'''