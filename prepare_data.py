import os
import shutil
import argparse
from io import BytesIO
import multiprocessing

import lmdb
from tqdm import tqdm
from PIL import Image
from torchvision import datasets


def convert(img, quality=100):
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()
    return val


def resize_worker(img_file, quality=100):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = convert(img, quality)
    return i, out


def prepare(env, imgs, n_worker):
    files = sorted(imgs)
    files = [(i, file) for i, file in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, img in tqdm(pool.imap_unordered(resize_worker, files)):
            key = f"{str(i).zfill(5)}".encode("utf-8")

            with env.begin(write=True) as txn:
                txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument(
        "--out",
        type=str,
        help="filename of the result lmdb dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "path",
        type=str,
        help="path to the image dataset",
    )

    args = parser.parse_args()

    imgs = map(
        lambda fn: os.path.join(args.path, fn),
        filter(
            lambda fn: any(
                map(
                    lambda ext: fn.endswith(ext),
                    datasets.folder.IMG_EXTENSIONS,
                )
            ),
            os.listdir(args.path),
        ),
    )

    shutil.rmtree(args.out)
    with lmdb.open(args.out, map_size=200 * 1024 ** 2, readahead=False) as env:
        prepare(env, imgs, args.n_worker)
