# Copyright 2022  Juhong Min
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# File based on HPF (https://github.com/juhongm999/hpf) with Apache License (above). 
# File adaped by Mohamed El Banani
import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASS_IDS = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "train": 19,
    "tvmonitor": 20,
}


class SPairDataset(torch.utils.data.Dataset):
    r"""Inherits CorrespondenceDataset"""

    def __init__(
        self,
        root,
        split,
        image_size=512,
        image_mean="imagenet",
        use_bbox=True,
        class_name=None,
        num_instances=None,
        vp_diff=None,
    ):
        """
        Constructs the SPair Dataset loader

        Inputs:
            root: Dataset root (where SPair is found; kinda odd TODO)
            thresh: how the threshold is calculated [img, bbox]
            split: dataset split to be used
            task: task for this dataset
        """
        super().__init__()
        assert split in ["train", "valid", "test"]

        self.root = root
        self.split = split
        self.image_size = image_size
        self.use_bbox = use_bbox

        if image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        elif image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError()

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

        instances = self.get_pair_annotations()

        if class_name:
            c_insts = [_a for _a in instances if _a["category"] == class_name]
            instances = c_insts

        if vp_diff is not None:
            instances = [_a for _a in instances if _a["viewpoint_variation"] == vp_diff]

        if num_instances:
            random.seed(20)
            random.shuffle(instances)
            instances = instances[:num_instances]

        self.instances = instances
        self.image_annotations = self.get_image_annotations()

    def process_keypoints(self, kp_dict, bbox, num_kps=None):
        num_kps = len(kp_dict) if num_kps is None else num_kps
        all_kps = [(kp_dict[str(i)], i) for i in range(num_kps) if kp_dict[str(i)]]
        kps_xy = torch.tensor([_xy for _xy, _ in all_kps]).int()
        kps_id = torch.tensor([_id for _, _id in all_kps]).long()

        if bbox:
            kps_xy[:, 0] -= bbox[0]
            kps_xy[:, 1] -= bbox[1]

        # generate full tensor
        kps = torch.zeros(num_kps, 3).int()
        kps[kps_id, :2] = kps_xy
        kps[kps_id, 2] = 1

        return kps

    def __getitem__(self, index, square=True):
        pair_i = self.instances[index]
        class_name = pair_i["category"]
        class_dict = self.image_annotations[class_name]
        _, view_i, view_j = pair_i["filename"].split(":")[0].split("-")

        # gett bounding boxes
        bbx_i = pair_i["src_bndbox"] if self.use_bbox else None
        bbx_j = pair_i["trg_bndbox"] if self.use_bbox else None

        kps_i = self.process_keypoints(class_dict[view_i]["kps"], bbx_i)
        kps_j = self.process_keypoints(class_dict[view_j]["kps"], bbx_j)

        img_i = self.get_image(class_name, view_i, bbox=bbx_i, square=square)
        seg_i = self.get_mask(class_name, view_i, bbox=bbx_i, square=square)
        img_j = self.get_image(class_name, view_j, bbx_j, square=square)
        seg_j = self.get_mask(class_name, view_j, bbox=bbx_j, square=square)

        # transform image
        hw_i = img_i.size[0]
        hw_j = img_j.size[0]

        if not self.use_bbox:
            l, u, r, d = pair_i["trg_bndbox"]
            max_bbox = max(r - l, d - u)
            max_idim = max(pair_i["trg_imsize"][:2])
            thresh_scale = float(max_bbox) / max_idim
        else:
            thresh_scale = 1.0

        # transform images
        img_i = self.image_transform(img_i)
        img_j = self.image_transform(img_j)
        seg_i = self.mask_transform(seg_i)
        seg_j = self.mask_transform(seg_j)
        kps_i[:, :2] = kps_i[:, :2] * self.image_size / hw_i
        kps_j[:, :2] = kps_j[:, :2] * self.image_size / hw_j

        return img_i, seg_i, kps_i, img_j, seg_j, kps_j, thresh_scale, class_name

    def __len__(self):
        return len(self.instances)

    def get_image(self, class_name, image_name, bbox=None, square=False):
        rel_path = f"JPEGImages/{class_name}/{image_name}.jpg"
        path = os.path.join(self.root, rel_path)
        with Image.open(path) as f:
            image = np.array(f)

        if bbox:
            l, u, r, d = bbox
            # if square:
            #     max_hw = max(d-u, r-l)
            #     h, w, _ = image.shape
            #     d = min(u + max_hw, h)
            #     r = min(l + max_hw, w)

            image = image[u:d, l:r]

        if square:
            h, w, _ = image.shape
            max_hw = max(h, w)
            image = np.pad(
                image, ((0, max_hw - h), (0, max_hw - w), (0, 0)), constant_values=255
            )

        return Image.fromarray(image)

    def get_mask(self, class_name, image_name, bbox=None, square=False):
        rel_path = f"Segmentation/{class_name}/{image_name}.png"
        path = os.path.join(self.root, rel_path)

        with Image.open(path) as img:
            image = np.array(img)

        if bbox:
            l, u, r, d = bbox
            image = image[u:d, l:r]

        if square:
            h, w = image.shape
            max_hw = max(h, w)
            image = np.pad(image, ((0, max_hw - h), (0, max_hw - w)))

        # big assumption of no other same class within bbox (or image)
        class_id = CLASS_IDS[class_name]
        image = (image == class_id).astype(float) * 255

        return Image.fromarray(image)

    def get_pair_annotations(self):
        split_names = {"train": "trn", "valid": "val", "test": "test"}
        split = split_names[self.split]

        annot_path = os.path.join(self.root, "PairAnnotation", split)
        annot_files = glob.glob(os.path.join(annot_path, "*.json"))
        annots = [json.load(open(_path)) for _path in annot_files]
        return annots

    def get_image_annotations(self):
        annot_path = os.path.join(self.root, "ImageAnnotation")
        classes = os.listdir(annot_path)

        image_annots = {_c: {} for _c in classes}

        for _cls in classes:
            annot_files = glob.glob(os.path.join(annot_path, f"{_cls}/*.json"))
            annots = [json.load(open(_path)) for _path in annot_files]
            annots = {_a["filename"].split(".")[0]: _a for _a in annots}
            image_annots[_cls] = annots

        return image_annots
