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
import numpy as np

keypoint_csv = np.loadtxt("spair_keypoint_names.csv", delimiter=",", dtype=str)
keypoint_csv = keypoint_csv.transpose()
class_names = [_cls.strip() for _cls in keypoint_csv[1:, 0]]

keypoint_csv = keypoint_csv[1:, 1:]
num_rows, num_cols = keypoint_csv.shape
kp_dict = {}

for row in range(num_rows):
    class_name = class_names[row]
    kp_dict[class_name] = {}

    for col in range(num_cols):
        kp_name = keypoint_csv[row, col].strip()
        if kp_name in ["", "N/A"]:
            continue
        kp_part = kp_name.split(" ")

        if len(kp_part) == 1:
            kp_dict[class_name][kp_name] = {"only": col}
        elif len(kp_part) == 2:
            _dir, _part = kp_part
            if _part in kp_dict[class_name]:
                kp_dict[class_name][_part][_dir] = col
            else:
                kp_dict[class_name][_part] = {_dir: col}
        else:
            print(f"How to parse {kp_name}?")

for class_name in class_names:
    print(f"========== {class_name:^20s} ==========")
    for kp in kp_dict[class_name]:
        # if len(kp_dict[class_name][kp]) == 1:
        #     try:
        #         _print = kp_dict[class_name][kp]['only']
        #     except:
        #         print(kp_dict[class_name][kp])
        #     print(f"{kp:10s}: {_print}")
        # else:
        _kp_dict = kp_dict[class_name][kp]
        _print = " \t ".join([f"{_dir:20s}: {_kp_dict[_dir]:2d}" for _dir in _kp_dict])
        print(f"{kp:20s}: {_print}")

breakpoint()
