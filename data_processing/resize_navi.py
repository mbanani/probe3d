"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import glob
from pathlib import Path

from joblib import Parallel, delayed
from PIL import Image, ImageOps
from tqdm import tqdm


def resize_image(path, interp=2, new_size=1024):
    """
    Resize each image in-place (rewrite the images)
    Input:
        path: image path (str)
        interp: interpolation type (0: nearest, 1: bilinear, 2: bicubic)
        new_size: minimum image dimension size
    """
    assert interp in [0, 1, 2]
    # do not resize already resize images
    if "downsampled_" in path.split("/")[-1]:
        return

    # Read image; need to exif_transpose some images to fit annotations
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image.copy()

    # resize so that smallest dimension is new_size
    width, height = image.size
    factor = float(new_size) / min(width, height)
    new_size = int(width * factor), int(height * factor)
    image = image.resize(new_size, interp)
    new_path = path.split("/")
    new_path[-1] = "downsampled_" + new_path[-1]

    new_path = "/".join(new_path)
    image.save(new_path)


if __name__ == "__main__":
    data_root = Path(__file__).parent / "../data/navi_v1"
    data_root = data_root.resolve()

    # get all image paths
    all_images = glob.glob(str(data_root / "*/*/images/*.jpg"))

    print("resize rgb images")
    Parallel(n_jobs=32)(delayed(resize_image)(path, 2) for path in tqdm(all_images))

    # get all image paths
    all_images = glob.glob(str(data_root / "*/*/depth/*.png"))
    print("resize depth images")
    Parallel(n_jobs=32)(delayed(resize_image)(path, 0) for path in tqdm(all_images))
