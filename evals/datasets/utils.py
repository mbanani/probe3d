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
import albumentations as A_transforms
import numpy as np
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as transform_F
from PIL import Image, ImageOps
from torch.linalg import cross
from torch.nn.functional import normalize


def get_navi_transforms(
    image_mean,
    image_size=None,
    augment=False,
    additional_targets=None,
    rotateflip=False,
    center_crop=True,
):
    if image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif image_mean == "None":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        raise ValueError()

    # get image transform -- use min_size to preserve aspect ratio
    image_transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
            tv_transforms.Resize(
                min(image_size), interpolation=tv_transforms.InterpolationMode.NEAREST
            ),
        ]
    )

    target_transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Resize(
                min(image_size), interpolation=tv_transforms.InterpolationMode.NEAREST
            ),
        ]
    )

    if center_crop:
        image_transform.transforms.append(tv_transforms.CenterCrop(min(image_size)))
        target_transform.transforms.append(tv_transforms.CenterCrop(min(image_size)))

    # parse dataset
    if augment:
        assert additional_targets is not None

        # insert before normalization
        image_transform.transforms.insert(
            -1,
            tv_transforms.RandomApply(
                [tv_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.8
            ),
        )

        # add shared transformations
        assert image_size is not None
        _h, _w = image_size
        # need to figure out rotation and flip for surface normals
        p_rotflip = 0.5 if rotateflip else 0.0
        shared_transform = A_transforms.Compose(
            [
                A_transforms.Resize(_h, _w, interpolation=0),
                A_transforms.Rotate(limit=10, interpolation=0, p=p_rotflip),
                A_transforms.HorizontalFlip(p=p_rotflip),
            ],
            additional_targets=additional_targets,
        )
    else:
        _h, _w = image_size
        shared_transform = A_transforms.Compose(
            [A_transforms.Resize(_h, _w, interpolation=0)],
            additional_targets=additional_targets,
        )

    return image_transform, target_transform, shared_transform


def get_nyu_transforms(
    image_mean, image_size=None, augment=False, additional_targets=None, rotateflip=True
):
    """
    Generates image and shared augmentation for NYU depth and surface normal prediction

    Input:
        image_mean (str): whether to use imagenet or clip mean
        augment (bool): whether or not to augment
        additional_targets (dict): dictionary for albumenation shared transform
    """

    if image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif image_mean == "None":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        raise ValueError()

    # get image transform
    image_transform = tv_transforms.Compose(
        [tv_transforms.ToTensor(), tv_transforms.Normalize(mean=mean, std=std)]
    )

    # parse dataset
    if augment:
        assert additional_targets is not None

        # insert before normalization
        image_transform.transforms.insert(
            -1,
            tv_transforms.RandomApply(
                [tv_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.8
            ),
        )

        # add shared transformations
        assert image_size is not None
        _h, _w = image_size
        p_rotflip = 0.5 if rotateflip else 0.0
        shared_transform = A_transforms.Compose(
            [
                A_transforms.HorizontalFlip(p=p_rotflip),
                A_transforms.Rotate(limit=10, interpolation=0, p=p_rotflip),
                A_transforms.RandomResizedCrop(
                    _h, _w, scale=(0.5, 1.0), ratio=(1.0, 1.0), p=0.5, interpolation=0
                ),
            ],
            additional_targets=additional_targets,
        )
    else:
        shared_transform = None

    return image_transform, shared_transform


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


def compute_normal(depth, focal_length, gaussian_blur=False):
    """
        Compute surface normals based on
    """
    # compute intrinsics
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length

    # compute masks
    mask = (depth > 0).float()
    if gaussian_blur:
        depth = transform_F.gaussian_blur(depth, 3, [1.0, 1.0])

    # if missing depth (depth = 0), make it very far
    depth[depth == 0] = 1e6

    # project depth intro xyz locations for intrinsics computation
    _, depth_h, depth_w = depth.shape
    grid = get_grid(depth_h, depth_w)
    xyd = grid * depth
    xyz = (torch.inverse(intrinsics) @ xyd.view(3, -1)).view(3, depth_h, depth_w)

    # compute partial deriviatives
    diff_l = xyz[:, 1:-1, :-2] - xyz[:, 1:-1, 1:-1]
    diff_t = xyz[:, :-2, 1:-1] - xyz[:, 1:-1, 1:-1]
    diff_r = xyz[:, 1:-1, 2:] - xyz[:, 1:-1, 1:-1]
    diff_b = xyz[:, 2:, 1:-1] - xyz[:, 1:-1, 1:-1]

    # compute normals based on multiple derivatives
    normal = torch.zeros_like(xyz)
    normal_lt = cross(diff_l, diff_t, dim=0)
    normal_tr = cross(diff_t, diff_r, dim=0)
    normal_rb = cross(diff_r, diff_b, dim=0)
    normal_bl = cross(diff_b, diff_l, dim=0)

    normal[:, 1:-1, 1:-1] = (normal_lt + normal_tr + normal_rb + normal_bl) / 4.0
    normal = normalize(normal, p=2, dim=0)
    normal = normal * mask
    return normal


def read_image(image_path: str, exif_transpose: bool = True) -> Image.Image:
    """Reads a NAVI image (and rotates it according to the metadata)."""
    with open(image_path, "rb") as f:
        with Image.open(f) as image:
            if exif_transpose:
                image = ImageOps.exif_transpose(image)
            image.convert("RGB")
            return image


def read_depth(path: str, scale_factor: float = 10.0) -> np.ndarray:
    depth_image = Image.open(path)

    max_val = (2 ** 16) - 1
    disparity = np.array(depth_image).astype("uint16")
    disparity = disparity.astype(np.float32) / (max_val * scale_factor)
    disparity[disparity == 0] = np.inf
    depth = 1 / disparity

    return depth


def bbox_crop(image, depth, xyz_grid):
    mask = depth > 0
    mask_coords = mask.nonzero()
    tl_coord = mask_coords.min(dim=0).values[1:]
    br_coord = mask_coords.max(dim=0).values[1:]

    # get sizes
    box_size = br_coord - tl_coord
    img_size = torch.tensor(mask.shape[1:])
    assert box_size.max() <= img_size.min(), "Aspect ratio prevents square crop"

    # figure out pad
    pad_size = box_size.max() - box_size
    tl_cent = tl_coord - pad_size // 2
    bl_cent = tl_cent + box_size.max()

    if (tl_cent >= 0).all() and (bl_cent <= img_size).all():
        image = image[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
        depth = depth[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
        xyz_grid = xyz_grid[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
    else:
        # get a far left corner and make sure it's in the picture
        tl_far = (tl_coord - pad_size).clip(min=0)
        bl_far = tl_far + box_size.max()

        image = image[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]
        depth = depth[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]
        xyz_grid = xyz_grid[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]

    return image, depth, xyz_grid


def rotate_crop(image, depth, xyz, angle):

    # rotate image and depth images
    image = transform_F.rotate(image, angle)
    depth = transform_F.rotate(depth, angle)
    xyz = transform_F.rotate(xyz, angle)

    # mask it
    mask = depth > 0
    mask_coords = mask.nonzero()
    tl_coord = mask_coords.min(dim=0).values[1:]
    br_coord = mask_coords.max(dim=0).values[1:]

    # get sizes
    box_size = br_coord - tl_coord
    img_size = torch.tensor(mask.shape[1:])
    assert box_size.max() <= img_size.min(), "Aspect ratio prevents square crop"

    # figure out pad
    pad_size = box_size.max() - box_size
    tl_cent = tl_coord - pad_size // 2
    bl_cent = tl_cent + box_size.max()

    if (tl_cent >= 0).all() and (bl_cent <= img_size).all():
        image = image[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
        depth = depth[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
        xyz = xyz[:, tl_cent[0] : bl_cent[0], tl_cent[1] : bl_cent[1]]
    else:
        # get a far left corner and make sure it's in the picture
        tl_far = (tl_coord - pad_size).clip(min=0)
        bl_far = tl_far + box_size.max()

        image = image[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]
        depth = depth[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]
        xyz = xyz[:, tl_far[0] : bl_far[0], tl_far[1] : bl_far[1]]

    return image, depth, xyz


def camera_matrices_from_annotation(annotation):
    """Convert camera pose and intrinsics to 4x4 matrices."""
    translation = translate(annotation["camera"]["t"])
    rotation = quaternion_to_rotation_matrix(annotation["camera"]["q"])
    object_to_world = translation @ rotation

    return object_to_world


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Computes a rotation matrix from a quaternion.

    Args:
        q: Rotation quaternions, float32[B*, 4]

    Returns:
        Rotation matrices, float32[B, 4, 4]

    """
    q = torch.as_tensor(q, dtype=torch.float32)
    w, x, y, z = torch.unbind(q, dim=-1)
    ZERO = torch.zeros_like(z)
    ONES = torch.ones_like(z)
    s = 2.0 / (q * q).sum(dim=-1)
    R_00 = 1 - s * (y ** 2 + z ** 2)
    R_01 = s * (x * y - z * w)
    R_02 = s * (x * z + y * w)
    R_10 = s * (x * y + z * w)
    R_11 = 1 - s * (x ** 2 + z ** 2)
    R_12 = s * (y * z - x * w)
    R_20 = s * (x * z - y * w)
    R_21 = s * (y * z + x * w)
    R_22 = 1 - s * (x ** 2 + y ** 2)

    rotation = torch.stack(
        [
            R_00,
            R_01,
            R_02,
            ZERO,
            R_10,
            R_11,
            R_12,
            ZERO,
            R_20,
            R_21,
            R_22,
            ZERO,
            ZERO,
            ZERO,
            ZERO,
            ONES,
        ],
        dim=-1,
    )

    return rotation.reshape(q.shape[:-1] + (4, 4))


def translate(v: torch.Tensor) -> torch.Tensor:
    """Computes a homogeneous translation matrices from translation vectors.

    Args:
      v: Translation vectors, `float32[B*, N]`

    Returns:
      Translation matrices, `float32[B*, N+1, N+1]`
    """
    result = torch.as_tensor(v, dtype=torch.float32)
    dimensions = result.shape[-1]
    result = result[..., None, :].transpose(-1, -2)
    result = torch.constant_pad_nd(result, [dimensions, 0, 0, 1])
    id_matrix = torch.diag(result.new_ones([dimensions + 1]))
    id_matrix = id_matrix.expand_as(result)
    result = result + id_matrix
    return result
