# Datasets

## NAVI 

[NAVI](https://github.com/google/navi) is a multi-view dataset that depicts 36 objects in a variety of scenes and poses.  Furthermore, the dataset comes with high-quality meshes and precise image-object alignment.  We use this dataset to evaluate the 3D awareness of models with a focus of objects. Please use the scripts below to download the dataset.  Additionally, we note that NAVI has very high resolution images, so we recommend first downsampling to avoid slow data loading. 

```bash
# Download and extract
cd data/
wget http://storage.googleapis.com/gresearch/navi-dataset/navi_v1.tar.gz
tar -xzf navi_v1.tar.gz

# downsample
cd ../data_processing
python resize_navi.py
```

## ScanNet Correspondence Test Split 

[ScanNet](https://www.scan-net.org/) is a large dataset of RGB-D video that depicts indoor scene.  We use the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) test split which samples 1500 image pairs for correspondence estimation. Please use the script below which downloads the split from the [LoFTR](https://github.com/zju3dv/LoFTR/) website.

```bash
# Move to data directory 
cd data/

# download the tar file provided by LoFTR
gdown --id 1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd
tar -xvf scannet_test_1500.tar
rm  scannet_test_1500.tar

cd scannet_test_1500
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/intrinsics.npz
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/test.npz 
```

## SPair-71k

The [SPair](https://cvlab.postech.ac.kr/research/SPair-71k/) dataset consists of image pair depicting instances of the same class with keypoint annotations as well as attributes of the image pair such as degree of viewpoint variation.  We additionally annotate the names of the keypoints to allow us to interpret the confusion matrices. 

```bash
# download and extract the dataset
cd data/
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xvf SPair-71k.tar.gz
rm SPair-71k.tar.gz
```


## NYU dataset 

The NYU dataset is one of the standard depth estimation benchmarks.  Furthermore, it is often used to evaluate surface normals using the annotations computed by [Ladicky et al.](https://inf.ethz.ch/personal/pomarc/pubs/LadickyECCV14.pdf). However, Ladicky only computed the surface normals for the labeled set of NYU which consists of 1449 images.  Following prior work, we use the labeled set's test set and Ladicky's annotations for evaluation.  However, we use a larger annotated version of the dataset annotated by [Bansal et al](https://github.com/aayushbansal/MarrRevisited) and released by [Qi et al](https://github.com/xjqi/GeoNet).

First, we download and extract the training set from GeoNet
```bash
cd data
# Download data1.zip and data.zip following https://github.com/xjqi/GeoNet
# the files are separated for simpler download, but contain two parts of the dataset
unzip data1.zip
unzip data2.zip
mkdir nyu_geonet
mv data1/* nyu_geonet/.
mv data2/* nyu_geonet/.
rmdir data1 data2
```

Next, we download and extract the test set from NYUv2 and the surface normal annotations from  FAIR's [SSL Benchmark](https://github.com/facebookresearch/fair_self_supervision_benchmark).
```bash
cd data
mkdir nyuv2
cd nyuv2

# download the original dataset from NYU and the surface normal annotations from Goyal et al. 
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip

# unzip and move up
unzip nyuv2_surfacenormal_metadata.zip
mv surfacenormal_metadata/* .
rm nyuv2_surfacenormal_metadata.zip 
rmdir surfacenormal_metadata

# compile everything into a single pkl for data loading
cd ../../data_processing
python create_nyu_pkl.py
```
