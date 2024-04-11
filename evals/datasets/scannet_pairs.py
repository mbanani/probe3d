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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as transforms

from .utils import read_image


class ScanNetPairsDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        # Some defaults for consistency.
        self.name = "ScanNet-pairs"
        self.root = "data/scannet_test_1500"
        self.split = "test"
        self.num_views = 2

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # parse files for data
        self.instances = self.get_instances(self.root)

        # Print out dataset stats
        print(f"{self.name} | {len(self.instances)} pairs")

    def get_dep(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = np.array(img)
                img = torch.tensor(img).float() / 1000.0
                return img[None, :, :]

    def get_instances(self, root_path):
        K_dict = dict(np.load(f"{root_path}/intrinsics.npz"))
        data = np.load(f"{root_path}/test.npz")["name"]
        instances = []

        for i in range(len(data)):
            room_id, seq_id, ins_0, ins_1 = data[i]
            scene_id = f"scene{room_id:04d}_{seq_id:02d}"
            K_i = torch.tensor(K_dict[scene_id]).float()

            instances.append((scene_id, ins_0, ins_1, K_i))

        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        s_id, ins_0, ins_1, K = self.instances[index]

        # paths
        rgb_path_0 = os.path.join(self.root, s_id, f"color/{ins_0}.jpg")
        rgb_path_1 = os.path.join(self.root, s_id, f"color/{ins_1}.jpg")
        dep_path_0 = os.path.join(self.root, s_id, f"depth/{ins_0}.png")
        dep_path_1 = os.path.join(self.root, s_id, f"depth/{ins_1}.png")

        # get rgb
        rgb_0 = read_image(rgb_path_0, exif_transpose=False)
        rgb_1 = read_image(rgb_path_1, exif_transpose=False)
        rgb_0 = self.rgb_transform(rgb_0)
        rgb_1 = self.rgb_transform(rgb_1)

        # get depths
        dep_0 = self.get_dep(dep_path_0)
        dep_1 = self.get_dep(dep_path_1)

        # get poses
        pose_path_0 = os.path.join(self.root, s_id, f"pose/{ins_0}.txt")
        pose_path_1 = os.path.join(self.root, s_id, f"pose/{ins_1}.txt")
        Rt_0 = torch.tensor(np.loadtxt(pose_path_0, delimiter=" "))
        Rt_1 = torch.tensor(np.loadtxt(pose_path_1, delimiter=" "))
        Rt_01 = Rt_1.inverse() @ Rt_0

        return {
            "uid": index,
            "class_id": "ScanNet_test",
            "sequence_id": s_id,
            "frame_0": int(ins_0),
            "frame_1": int(ins_1),
            "K": K,
            "rgb_0": rgb_0,
            "rgb_1": rgb_1,
            "depth_0": dep_0,
            "depth_1": dep_1,
            "Rt_0": torch.eye(4).float(),
            "Rt_1": Rt_01.float(),
        }
