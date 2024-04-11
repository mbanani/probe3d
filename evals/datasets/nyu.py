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
import os
import pickle

import numpy as np
import scipy
import torch

from .utils import get_nyu_transforms


def NYU(
    train_path,
    test_path,
    split,
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
):
    assert split in ["train", "trainval", "valid", "test"]
    if split == "test":
        return NYU_test(test_path, image_mean, center_crop)
    else:
        return NYU_geonet(
            train_path,
            split,
            image_mean,
            center_crop,
            augment_train,
            rotateflip=rotateflip,
        )


class NYU_test(torch.utils.data.Dataset):
    """
    Dataset loader based on Ishan Misra's SSL benchmark
    """

    def __init__(self, path, image_mean="imagenet", center_crop=False):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            False,
            rotateflip=False,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        # parse data
        with open(path, "rb") as f:
            data_dict = pickle.load(f)

        self.indices = data_dict["test_indices"]
        self.depths = [data_dict["depths"][_i] for _i in self.indices]
        self.images = [data_dict["images"][_i] for _i in self.indices]
        self.scenes = [data_dict["scene_types"][_i][0] for _i in self.indices]
        self.snorms = [data_dict["snorms"][_i] for _i in self.indices]

        num_instances = len(self.indices)
        print(f"NYUv2 labeled test set: {num_instances} instances")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depths[index]
        snorm = self.snorms[index]
        room = self.scenes[index]
        nyu_index = self.indices[index]

        # transform image
        image = np.transpose(image, (1, 2, 0))
        image = self.image_transform(image)

        # set max depth to 10
        depth[depth > 10] = 0

        # center crop
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]

        # move to tensor
        depth = torch.tensor(depth).float()[None, :, :]
        snorm = torch.tensor(snorm).float()

        return {
            "image": image,
            "depth": depth,
            "snorm": snorm,
            "room": room,
            "nyu_index": nyu_index,
        }


class NYU_geonet(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split,
        image_mean="imagenet",
        center_crop=False,
        augment_train=False,
        rotateflip=False,
    ):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        augment = augment_train and "train" in split
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            augment,
            rotateflip=rotateflip,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        # parse dataset
        self.root_dir = path
        insts = os.listdir(path)
        insts.sort()

        # remove bad indices
        del insts[21181]
        del insts[6919]

        assert split in ["train", "valid", "trainval"]
        if split == "train":
            self.instances = [x for i, x in enumerate(insts) if i % 20 != 0]
        elif split == "valid":
            self.instances = [x for i, x in enumerate(insts) if i % 20 == 0]
        elif split == "trainval":
            self.instances = insts
        else:
            raise ValueError()

        print(f"NYU-GeoNet {split}: {len(self.instances)} instances.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        file_name = self.instances[index]
        room = "_".join(file_name.split("-")[0].split("_")[:-2])

        # extract elements from the matlab thing
        instance = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        image = instance["img"][:480, :640]
        depth = instance["depth"][:480, :640]
        snorm = torch.tensor(instance["norm"][:480, :640]).permute(2, 0, 1)

        # process image
        image[:, :, 0] = image[:, :, 0] + 2 * 122.175
        image[:, :, 1] = image[:, :, 1] + 2 * 116.169
        image[:, :, 2] = image[:, :, 2] + 2 * 103.508
        image = image.astype(np.uint8)
        image = self.image_transform(image)

        # set max depth to 10
        depth[depth > self.max_depth] = 0

        # center crop
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]

        if self.shared_transform:
            # put in correct format (h, w, feat)
            image = image.permute(1, 2, 0).numpy()
            snorm = snorm.permute(1, 2, 0).numpy()
            depth = depth[:, :, None]

            # transform
            transformed = self.shared_transform(image=image, depth=depth, snorm=snorm)

            # get back in (feat_dim x height x width)
            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
        else:
            # move to torch tensors
            depth = torch.tensor(depth).float()[None, :, :]
            snorm = torch.tensor(snorm).float()

        return {"image": image, "depth": depth, "snorm": snorm, "room": room}

    def split_trainval(self, split, indices, scene_types):
        """
        Parses the directory for instances.
        Input: data_dict -- sturcture  <object_id>/<collection>/<instances>

        Output: all dataset instances
        """
        # get instances
        assert split in ["train", "valid"]

        # get all image collections by room type
        split_dict = {}
        for ind in indices:
            scene = scene_types[ind][0]
            if scene not in split_dict:
                split_dict[scene] = [ind]
            else:
                split_dict[scene].append(ind)

        # split images based on room type
        train_ins, valid_ins = [], []
        for scene in split_dict:
            scene_inds = split_dict[scene]
            if len(scene_inds) < 5:
                train_ins += scene_inds
            else:
                scene_inds.sort()
                ratio = int(0.80 * len(scene_inds))
                train_ins += scene_inds[:ratio]
                valid_ins += scene_inds[ratio:]

        # collect instances based on correct coll
        insts = train_ins if split == "train" else valid_ins
        return insts
