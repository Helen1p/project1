import os
from glob import glob
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm


def mkdirs_s64(root_dir, root_dir_s64):
    for sub_dir in sub_dirs:
        dir_paths = glob(os.path.join(root_dir, "*", sub_dir))
        for dir_path in dir_paths:
            dir_path_s64 = dir_path.replace(root_dir, root_dir_s64)
            os.makedirs(dir_path_s64, exist_ok=True)


def resize_dir_s64(root_dir, root_dir_s64, split):
    for idx, sub_dir in enumerate(sub_dirs):
        print(f"Handling {split}, dir: {idx + 1}/{len(sub_dirs)}")

        img_paths = glob(os.path.join(root_dir, "*", sub_dir, "*.png"))
        img_paths_s64 = [_.replace(root_dir, root_dir_s64) for _ in img_paths]

        pbar = tqdm(total=len(img_paths), unit='image', desc='Resize')
        pool = Pool(num_workers)
        for img_path, img_path_s64 in zip(img_paths, img_paths_s64):
            pool.apply_async(worker, args=(img_path, img_path_s64), callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()
        pbar.close()


def worker(img_path, img_path_s64):
    img = Image.open(img_path)
    img_s64 = img.resize(size, Image.BICUBIC)
    img_s64.save(img_path_s64)


if __name__ == "__main__":
    num_workers = 20
    size = (64, 64)
    sub_dirs = ["ref", "p0", "p1"]
    train_dir = "twoafc_train/train"
    val_dir = "twoafc_val/val"
    train_dir_s64 = "twoafc_train_s64/train"
    val_dir_s64 = "twoafc_val_s64/val"

    # make dirs to save resized 64 x 64 images
    mkdirs_s64(train_dir, train_dir_s64)
    mkdirs_s64(val_dir, val_dir_s64)

    # resize images to 64 x 64
    resize_dir_s64(train_dir, train_dir_s64, split="train")
    resize_dir_s64(val_dir, val_dir_s64, split="val")
