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
import json
import os
from pathlib import Path

import numpy as np
import torch

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
)


class NAVI(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        model="all",
        image_mean="imagenet",
        augment_train=False,
        rotateflip=False,
        bbox_crop=True,
        pair_dataset=False,
        max_angle=120,
        relative_depth=False,
    ):
        super().__init__()

        # generate splits based on collections and subparts
        if split == "train":
            collection = "multiview"
            subpart = "train"
        elif split == "valid":
            collection = "multiview"
            subpart = "test"
        elif split == "trainval":
            collection = "multiview"
            subpart = "all"
        elif split == "test":
            collection = "wild"
            subpart = "all"
        else:
            raise ValueError(f"Unknown split: {split}")

        # set path
        self.data_root = Path(path)
        self.bbox_crop = bbox_crop
        self.relative_depth = relative_depth
        self.max_depth = 1.0

        self.name = f"NAVI_{collection}_{subpart}"
        if relative_depth:
            self.name = self.name + "_reldepth"

        # parse dataset
        self.data_dict = self.parse_dataset()
        self.define_instances_split(model, collection, subpart)

        # get transforms
        augment = augment_train and "train" in split
        image_size = (512, 512)
        t_fns = get_navi_transforms(
            image_mean,
            image_size=image_size,
            augment=augment,
            rotateflip=rotateflip,
            additional_targets={
                "depth": "image",
                "snorm": "image",
                "xyz_grid": "image",
            },
        )
        self.image_transform, self.target_transform, self.shared_transform = t_fns
        print(f"NAVI {collection} {subpart} {model}: {len(self.instances)} instances")

        self.pair_dataset = pair_dataset
        self.max_angle = max_angle
        if self.pair_dataset:
            self.pair_indices = self.generate_instance_pairs(self.instances)

        self.instances = self.instances[::4]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        if self.pair_dataset:
            obj_id, scene_id, img_id_0 = self.instances[index]
            img_id_1 = self.pair_indices[obj_id][scene_id][img_id_0]

            inst_0 = self.get_single(obj_id, scene_id, img_id_0)
            inst_1 = self.get_single(obj_id, scene_id, img_id_1)

            output = {}
            for key in inst_0:
                output[f"{key}_0"] = inst_0[key]
                output[f"{key}_1"] = inst_1[key]

            # computer relative pose
            Rt_01 = output["Rt_1"] @ output["Rt_0"].inverse()
            output["Rt_01"] = Rt_01
            output["pair_id"] = f"{img_id_0}-{img_id_1}"
        else:
            obj_id, scene_id, img_id = self.instances[index]
            output = self.get_single(obj_id, scene_id, img_id)

        return output

    def get_single(self, obj_id, scene_id, img_id):
        obj_number = self.objects[obj_id]
        anno = self.data_dict[obj_id][scene_id]["annotations"][img_id]

        # get scene path
        prefix = "downsampled_"
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{prefix}{img_id}.jpg"
        depth_path = scene_path / f"depth/{prefix}{img_id}.png"

        # get image
        image = read_image(image_path)
        image = self.image_transform(image)

        # get depth -- move from millimeter to meters
        depth = read_depth(str(depth_path)) / 1000
        min_depth = depth[depth > 0].min()
        depth = self.target_transform(depth)

        #  === construct xyz at full image size and apply all transformations ===
        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)

        # intrnsics for augmented image
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = 0.5 * image_h  # assume offset is at the center
        intrinsics[1, 2] = 0.5 * image_w  # assume offset is at the center

        # make grid
        grid = get_grid(image_h, image_w)
        uvd_grid = depth * grid
        xyz = intrinsics.inverse() @ uvd_grid.view(3, image_h * image_w)
        xyz_grid = xyz.view(3, image_h, image_w)

        if self.bbox_crop:
            image, depth, xyz_grid = bbox_crop(image, depth, xyz_grid)

        bbox_h, bbox_w = image.shape[1:]
        snorm = compute_normal(depth.clone(), aug_fx)

        if self.shared_transform is not None:
            transformed = self.shared_transform(
                image=image.permute(1, 2, 0).numpy(),
                depth=depth.permute(1, 2, 0).numpy(),
                snorm=snorm.permute(1, 2, 0).numpy(),
                xyz_grid=xyz_grid.permute(1, 2, 0).numpy(),
            )

            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            xyz_grid = torch.tensor(transformed["xyz_grid"]).float().permute(2, 0, 1)

        # -- use min() to handle center cropping
        final_h, final_w = image.shape[1:]
        final_fx = aug_fx * min(final_h, final_w) / min(bbox_h, bbox_w)
        intrinsics = torch.eye(3)
        intrinsics[:2] = final_fx * intrinsics[:2]

        # remove weird depth artifacts from averaging
        depth[depth < min_depth] = 0

        # extract pose and change translation unit from mm to meters
        Rt = camera_matrices_from_annotation(anno)
        Rt[:3, 3] = Rt[:3, 3] / 1000.0

        if self.relative_depth:
            max_depth = depth.max()
            zero_mask = depth == 0

            # normalize depth between 0 and 1
            depth = (depth - min_depth) / max(0.01, max_depth - min_depth)
            depth = depth * 0.99 + 0.01

            # set zero deoth to zero
            depth[zero_mask] = 0

        # mask = (depth > 0).float()
        # image = (image * mask) + torch.ones_like(image) * (1 - mask)

        return {
            "image": image,
            "depth": depth,
            "class_id": obj_number,
            "intrinsics": intrinsics,
            "snorm": snorm,
            "Rt": Rt,
            "anno": anno,
            "xyz_grid": xyz_grid,
        }

    def parse_dataset(self):
        """
        Parses the directory for instances.
        Input: data_dict -- sturcture  <object_id>/<collection>/<instances>

        Output: all dataset instances
        """
        data_dict = {}

        # get all image folders
        all_collections = []
        all_collections += glob.glob(str(self.data_root / "*/multiview_*"))
        all_collections += glob.glob(str(self.data_root / "*/wild_set"))

        # parse image folders
        for collection_path in all_collections:
            object_id, collection_id = collection_path.split("/")[-2:]

            # get image ids
            img_files = os.listdir(os.path.join(collection_path, "images"))
            img_ids = [_file.split(".")[0] for _file in img_files if "jpg" in _file]

            # VERY hacky | remove small_
            img_ids = [_id for _id in img_ids if "_" not in _id]

            # load annotations and convert them to a dictionary
            with open(os.path.join(collection_path, "annotations.json")) as f:
                annotations = json.load(f)
                annotations = {
                    _an["filename"].split(".")[0]: _an for _an in annotations
                }

            # update data_dict
            if object_id not in data_dict:
                data_dict[object_id] = {}

            data_dict[object_id][collection_id] = {
                "views": img_ids,
                "annotations": annotations,
            }

        return data_dict

    def define_instances_split(self, model, collection, subpart):
        # get a list of object names
        if model == "all":
            object_names = list(self.data_dict.keys())
        else:
            object_names = [model]

        assert collection in ["multiview", "wild"]
        assert subpart in ["train", "test", "all"]

        self.instances = []
        self.objects = []

        for obj_id in object_names:
            scenes = list(self.data_dict[obj_id].keys())
            if "wild_set" not in scenes or len(scenes) == 1:
                print(f"Skipping object {obj_id}.")
                continue
            else:
                self.objects.append(obj_id)

            if collection == "wild":
                image_ids = self.data_dict[obj_id]["wild_set"]["views"]
                image_ann = self.data_dict[obj_id]["wild_set"]["annotations"]
                assert len(image_ids) > 1

                for _id in image_ids:
                    if subpart == "all":
                        self.instances.append((obj_id, "wild_set", _id))
                    else:
                        im_split = image_ann[_id]["split"]
                        if subpart == "train" and im_split == "train":
                            self.instances.append((obj_id, "wild_set", _id))
                        elif subpart == "test" and im_split == "val":
                            self.instances.append((obj_id, "wild_set", _id))
            else:
                scenes = [_s for _s in scenes if "multiview" in _s]

                # int floors -> at least 1 validation scene
                train_split = int(0.9 * len(scenes))

                if subpart == "train":
                    scenes = scenes[:train_split]
                elif subpart == "test":
                    scenes = scenes[train_split:]
                elif subpart == "all":
                    scenes = scenes
                else:
                    assert collection == "multiview", f"collection was {collection}."

                if len(scenes) == 0:
                    continue

                for scene in scenes:
                    image_ids = self.data_dict[obj_id][scene]["views"]
                    for _id in image_ids:
                        self.instances.append((obj_id, scene, _id))

        # create object -> class mapping
        self.objects.sort()
        self.objects = {_obj: _id for _id, _obj in enumerate(self.objects)}

    def generate_instance_pairs(self, instances):
        torch.manual_seed(8)
        inst_dict = {}
        for ins in instances:
            obj_id, coll_id, img_id = ins
            if obj_id not in inst_dict:
                inst_dict[obj_id] = {coll_id: [img_id]}
            elif coll_id not in inst_dict[obj_id]:
                inst_dict[obj_id][coll_id] = [img_id]
            else:
                inst_dict[obj_id][coll_id].append(img_id)

        pair_dict = {}
        for obj_id in inst_dict:
            pair_dict[obj_id] = {}
            for col_id in inst_dict[obj_id]:
                pair_dict[obj_id][col_id] = {}
                rots = []
                img_ids = []
                for img_id in inst_dict[obj_id][col_id]:
                    anno = self.data_dict[obj_id][col_id]["annotations"][img_id]
                    Rt = camera_matrices_from_annotation(anno)
                    rots.append(Rt[:3, :3])
                    img_ids.append(img_id)
                rots = torch.stack(rots, dim=0)

                # for each image find a pair between 0 and max_angle degrees
                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    rots_i = rots[i, None].repeat(len(rots), 1, 1)
                    rots_ij = torch.bmm(rots_i, rots.permute(0, 2, 1))
                    rots_tr = rots_ij[:, 0, 0] + rots_ij[:, 1, 1] + rots_ij[:, 2, 2]
                    rel_ang_rad = (0.5 * rots_tr - 0.5).clamp(min=-1, max=1).acos()
                    rel_ang_deg = rel_ang_rad * 180 / np.pi

                    # only sample from deg(0, max_angle)
                    rel_ang_deg[rel_ang_deg > self.max_angle] = 0
                    rel_ang_deg[rel_ang_deg > 0] = 1

                    # sample an element
                    pair_i = torch.multinomial(rel_ang_deg, 1).item()
                    pair_dict[obj_id][col_id][img_id] = img_ids[pair_i]

        return pair_dict
