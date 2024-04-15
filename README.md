Probing the 3D Awareness of Visual Foundation Models
=====================================================

This repository contains a re-implementation of the code for the paper [Probing the 3D Awareness of
Visual Foundation Models](https://arxiv.org/abs/2404.08636) (CVPR 2024) which presents an analysis of the 3D awareness of visual
foundation models.


[Mohamed El Banani](mbanani.github.io), [Amit Raj](https://amitraj93.github.io/), [Kevis-Kokitsi Maninis](https://www.kmaninis.com/), [Abhishek Kar](https://abhishekkar.info/), [Yuanzhen Li](https://people.csail.mit.edu/yzli/), [Michael Rubinstein](https://people.csail.mit.edu/mrub/), [Deqing Sun](https://deqings.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Justin Johnson](https://web.eecs.umich.edu/~justincj/),  [Varun Jampani](https://varunjampani.github.io/) 

If you find this code useful, please consider citing:  
```text
@inProceedings{elbanani2024probing,
  title={{Probing the 3D Awareness of Visual Foundation Models}},
  author={
        El Banani, Mohamed and Raj, Amit and Maninis, Kevis-Kokitsi and 
        Kar, Abhishek and Li, Yuanzhen and Rubinstein, Michael and Sun, Deqing and 
        Guibas, Leonidas and Johnson, Justin and Jampani, Varun
        },
  booktitle={CVPR},
  year={2024},
}
```

Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```bash
conda create -n probe3d python=3.9 --yes
conda activate probe3d
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -c conda-forge nb_conda_kernels=2.3.1

pip install -r requirements.txt
python setup.py develop

pip install protobuf==3.20.3    # weird dependency with datasets and google's api
pre-commit install              # install pre-commit
```


Finally, please follow the dataset download and preprocessing instructions [here](./data_processing/README.md).


Evaluation Experiments
-----------

We provide code to train the depth probes and evaluate the correspondence. All experiments use
hydra configs which can be found [here](./configs). Below are example commands for running the
evaluations with the DINO ViT-B/16 backbone.

```python
# Training single-view probes
python train_depth.py backbone=dino_b16 +backbone.return_multilayer=True
python train_snorm.py backbone=dino_b16 +backbone.return_multilayer=True

# Evaluating multiview correspondence 
python evaluate_navi_correspondence.py +backbone=dino_b16
python evaluate_scannet_correspondence.py +backbone=dino_b16
python evaluate_spair_correspondence.py +backbone=dino_b16
```


Performance Correlation
-----------

Coming soon.


Acknowledgments
-----------------

We thank Prafull Sharma, Shivam Duggal, Karan Desai, Junhwa Hur, and Charles Herrmann for many helpful discussions.
We also thank Alyosha Efros, David Fouhey, Stella Yu, and Andrew Owens for their feedback. 


We would also like to acknowledge the following repositories and users for releasing very valuable
code and datasets: 

- [GeoNet](https://github.com/xjqi/GeoNet) for releasing the extracted surface normals for full NYU.  
