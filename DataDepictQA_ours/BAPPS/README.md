---
license: apache-2.0
task_categories:
- image-to-text
language:
- en
tags:
- multi-modal image quality assessment
pretty_name: DataDepictQA
size_categories:
- 100K<n<1M
---

# DataDepictQA

Datasets of the papers in [DepictQA project](https://depictqa.github.io/):

- DepictQA-Wild (DepictQA-v2): [paper](https://arxiv.org/abs/2405.18842) / [project page](https://depictqa.github.io/depictqa-wild/) / [code](https://github.com/XPixelGroup/DepictQA).

  Zhiyuan You, Jinjin Gu, Zheyuan Li, Xin Cai, Kaiwen Zhu, Chao Dong, Tianfan Xue, "Descriptive Image Quality Assessment in the Wild," arXiv preprint arXiv:2405.18842, 2024.

- DepictQA-v1: [paper](https://arxiv.org/abs/2312.08962) / [project page](https://depictqa.github.io/depictqa-v1/) / [code](https://github.com/XPixelGroup/DepictQA).

  Zhiyuan You, Zheyuan Li, Jinjin Gu, Zhenfei Yin, Tianfan Xue, Chao Dong, "Depicting beyond scores: Advancing image quality assessment through multi-modal language models," ECCV, 2024.


## Dataset Overview

- Training DepictQA-v2 requires:
  - KADIS700K
  - BAPPS
  - PIPAL
  - KADID10K
  - DetailDescriptionLAMM

- Training DepictQA-v1 requires:
  - BAPPS
  - PIPAL
  - KADID10K
  - DetailDescriptionLAMM


## Dataset Construction

**Source codes** for dataset construction are provided in [here](https://github.com/XPixelGroup/DepictQA/tree/main/build_datasets). 

Our datasets are constructed based on existing datasets. Therefore, some source images should be downloaded and re-arranged to construct the datasets. Bellow we provide a detailed instruction.


### KADIS700K

1. Download our constructed dataset from [here](https://modelscope.cn/datasets/zhiyuanyou/DataDepictQA) (under the **KADIS700K** directory).
2. Place the downloaded images in `DataDepictQA/KADIS700K` as follows.
3. The meanings of directory names can be found in **Abbreviations** section of our [source codes](https://github.com/XPixelGroup/DepictQA/tree/main/build_datasets) for dataset construction.

```
|-- DataDepictQA
  |-- KADIS700K
    |-- A_md_brief
    |-- A_md_detail
    |-- A_sd_brief
    |-- A_sd_detail
    |-- AB_md_detail
    |-- AB_sd_detail
    |-- metas_combine
    |-- ref_imgs_s224 (downloaded)
    |-- refA_md_brief
      ｜-- dist_imgs (downloaded)
      ｜-- metas
    |-- refA_md_detail
      |-- dist_imgs (downloaded)
      |-- dist_imgs_test100 (downloaded)
      |-- metas
    |-- refA_sd_brief
      |-- dist_imgs (downloaded)
      |-- metas
    |-- refA_sd_detail
      |-- dist_imgs (downloaded)
      |-- dist_imgs_test200 (downloaded)
      |-- metas
    |-- refAB_md_detail
      |-- dist_imgs (downloaded)
      |-- dist_imgs_test100 (downloaded)
      |-- metas
    |-- refAB_sd_detail
      |-- dist_imgs (downloaded)
      |-- dist_imgs_test200 (downloaded)
      |-- metas
```


### BAPPS

1. Download the BAPPS dataset (**2AFC Train set** and **2AFC Val set**) from [here](https://github.com/richzhang/PerceptualSimilarity/blob/master/scripts/download_dataset.sh).
2. Place the downloaded images in `DataDepictQA/BAPPS` as follows.
```
|-- DataDepictQA
  |-- BAPPS
    |-- images
      |-- mbapps_test_refA_s64
      |-- mbapps_test_refAB_s64
      |-- twoafc_train (downloaded)
      |-- twoafc_val (downloaded)
      |-- resize_bapps.py
    |-- metas
```
3. The downloaded images are 256 x 256 patches, which are resized from the original 64 x 64 patches. 
   Resizing does not influence comparison results (_i.e._, Image A or Image B is better), but influences the detailed reasoning tasks since additional pixelation distortion is introduced.
   Therefore, we resize these images back to their original 64 x 64 resolution.
```
cd DataDepictQA/BAPPS/images
python resize_bapps.py
```
4. The constructed BAPPS directory should be as follows.
```
|-- DataDepictQA
  |-- BAPPS
    |-- images
      |-- mbapps_test_refA_s64
      |-- mbapps_test_refAB_s64
      |-- twoafc_train (downloaded)
      |-- twoafc_train_s64 (created by resize_bapps.py)
      |-- twoafc_val (downloaded)
      |-- twoafc_val_s64 (created by resize_bapps.py)
      |-- resize_bapps.py
    |-- metas
```


### PIPAL

1. Download the PIPAL dataset (**train set**) from [here](https://github.com/HaomingCai/PIPAL-dataset).
2. Place the downloaded images in `DataDepictQA/PIPAL` as follows.
```
|-- DataDepictQA
  |-- PIPAL
    |-- images
      |-- Distortion_1 (downloaded)
      |-- Distortion_2 (downloaded)
      |-- Distortion_3 (downloaded)
      |-- Distortion_4 (downloaded)
      |-- Train_Ref (downloaded)
    |-- metas
```


### KADID10K

1. Download the KADID10K dataset from [here](https://database.mmsp-kn.de/kadid-10k-database.html).
2. Place the downloaded images in `DataDepictQA/KADID10K` as follows.
```
|-- DataDepictQA
  |-- KADID10K
    |-- images (downloaded)
    |-- metas
```


### DetailDescriptionLAMM

1. Download the LAMM Detailed Description dataset (**coco_images**) from [here](https://opendatalab.com/LAMM/LAMM/tree/main/raw/2D_Instruct).
2. Place the downloaded images in `DataDepictQA/DetailDescriptionLAMM` as follows.
```
|-- DataDepictQA
  |-- DetailDescriptionLAMM
    |-- coco_images (downloaded)
    |-- metas
```

