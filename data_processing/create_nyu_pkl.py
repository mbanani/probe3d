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
import gzip
import json
import pickle

import mat73
import numpy as np

# load original dicts
nyuv2_dict = mat73.loadmat("../data/nyuv2/nyu_depth_v2_labeled.mat")
with gzip.GzipFile("../data/nyuv2/all_normals.pklz", "r") as file:
    snorm_dict = pickle.load(file)

# get images and snorm
all_depths = np.transpose(nyuv2_dict["rawDepths"], (2, 0, 1))
all_images = np.transpose(nyuv2_dict["images"], (3, 2, 0, 1))
all_snorms = np.transpose(snorm_dict["all_normals"], (0, 3, 1, 2))
scene_types = nyuv2_dict["sceneTypes"]

# get split data
train_json = json.load(open("../data/nyuv2/train_SN40.json"))
test_json = json.load(open("../data/nyuv2/test_SN40.json"))
train_split = [int(_ins["img"].split("_")[0]) - 1 for _ins in train_json]
test_split = [int(_ins["img"].split("_")[0]) - 1 for _ins in test_json]

# Save dictionary
save_dict = {
    "depths": np.transpose(nyuv2_dict["rawDepths"], (2, 0, 1)),
    "images": np.transpose(nyuv2_dict["images"], (3, 2, 0, 1)),
    "snorms": np.transpose(snorm_dict["all_normals"], (0, 3, 1, 2)),
    "scene_types": nyuv2_dict["sceneTypes"],
    "train_indices": np.array(train_split),
    "test_indices": np.array(test_split),
}

save_path = "../data/nyuv2/nyuv2_snorm_all.pkl"
print(f"Saving combined pkl file at {save_path}")
with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
