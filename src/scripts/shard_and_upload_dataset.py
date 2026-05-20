#!/usr/bin/env python3
"""
Shard the garment-particle dataset into tar archives and upload them to a
HuggingFace *dataset* repo.

Only the per-garment particle data is included: `garment_particles_rand_*.h5`
and `stats.txt`. GarmentCodeData-v2 is NOT included — users download that
separately from the official link.

Each shard tar preserves the `rand_<id>/<file>` layout, so a user downloads all
shards, extracts them into a single directory, and points the dataset config's
`data_dir` at it:

    huggingface-cli download georgeNakayama/GarmentParticles --repo-type dataset \\
        --local-dir garment_particles
    cd garment_particles && for t in data/*.tar; do tar -xf "$t"; done

Usage:
    # full run (build + upload, ~33 GB)
    python shard_and_upload_dataset.py --data-dir /path/to/garment_particles_v2.1_11182025

    # build shards locally only, no upload
    python shard_and_upload_dataset.py --data-dir ... --no-upload

    # quick test on a handful of garments
    python shard_and_upload_dataset.py --data-dir ... --limit 20 --shard-size 10 --no-upload
"""
import argparse
import os
import sys
import tarfile

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True,
                    help="the garment_particles_v2.1_11182025 directory")
    ap.add_argument("--repo-id", default="georgeNakayama/GarmentParticles",
                    help="target HuggingFace dataset repo")
    ap.add_argument("--shard-size", type=int, default=5000,
                    help="garments per shard (default 5000 -> ~1.3 GB shards)")
    ap.add_argument("--staging", default="./gp_shards",
                    help="scratch directory for tar shards")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N garments (0 = all; for testing)")
    ap.add_argument("--no-upload", action="store_true",
                    help="build shards locally, do not upload")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep tar shards after upload (default: delete each after upload)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        sys.exit(f"data-dir not found: {data_dir}")

    # one directory per garment: rand_<id>/  (skip non-rand entries / stray files)
    garments = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("rand_") and os.path.isdir(os.path.join(data_dir, d))
    )
    if args.limit:
        garments = garments[:args.limit]
    if not garments:
        sys.exit("no rand_* garment directories found in data-dir")
    print(f"{len(garments)} garments from {data_dir}")

    os.makedirs(args.staging, exist_ok=True)
    api = None
    if not args.no_upload:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    n_shards = (len(garments) + args.shard_size - 1) // args.shard_size
    print(f"{n_shards} shard(s), up to {args.shard_size} garments each\n")

    for i in range(n_shards):
        chunk = garments[i * args.shard_size:(i + 1) * args.shard_size]
        shard = os.path.join(args.staging, f"particles-{i:04d}.tar")
        n_files = 0
        with tarfile.open(shard, "w") as tar:
            for g in chunk:
                gdir = os.path.join(data_dir, g)
                for fn in sorted(os.listdir(gdir)):
                    # particle data only: per-garment .h5 + its stats.txt
                    if fn.endswith(".h5") or fn == "stats.txt":
                        tar.add(os.path.join(gdir, fn), arcname=f"{g}/{fn}")
                        n_files += 1
        size_gb = os.path.getsize(shard) / 1e9
        print(f"[{i + 1}/{n_shards}] built {os.path.basename(shard)}  "
              f"{len(chunk)} garments, {n_files} files, {size_gb:.2f} GB")

        if api is not None:
            api.upload_file(
                repo_id=args.repo_id, repo_type="dataset",
                path_or_fileobj=shard,
                path_in_repo=f"data/{os.path.basename(shard)}",
                commit_message=f"Add particle shard {i:04d}",
            )
            print(f"           uploaded -> data/{os.path.basename(shard)}")
            if not args.keep_shards:
                os.remove(shard)

    if args.no_upload:
        print(f"\nDONE — shards in {args.staging} (not uploaded)")
    else:
        print(f"\nALL DONE — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
