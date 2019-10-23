# kuzushiji-recognition-2019

[https://www.kaggle.com/c/kuzushiji-recognition/](https://www.kaggle.com/c/kuzushiji-recognition/)

3rd place solution

See [overview.md](overview.md) for an overview of my solution.


## Prerequisites

- python 3.6 or 3.7
- docker
- nvidia-docker

## Setup

```
$ pip install -r requirements.txt
```

## Download and preprocess dataset of this competition

Download and place the dataset of this competition as below.

```
input
├── train_images/
├── test_images/
└── train.csv
```

and run `preprocess.sh`

```
$ bash preprocess.sh
```


## Reproduce final submission

### Download trained models

Download the following 8 models and save it in the `models` directory

#### detection models

- [model_0060000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/model_0060000.pth)
- [model_0100000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/model_0100000.pth)

#### classification models

- [01_refine_efficientnet_b4_l2softmax_gray190-0060.model](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/01_refine_efficientnet_b4_l2softmax_gray190-0060.model)
- [02_refine_resnet152_l2softmax_gray112-0069.model](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/02_refine_resnet152_l2softmax_gray112-0069.model)
- [03_refine_seresnext101_l2softmax_rgb112-0080.model](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/03_refine_seresnext101_l2softmax_rgb112-0080.model)
- [04_refine_seresnext101_l2softmax_rgb112-0082.model](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/04_refine_seresnext101_l2softmax_rgb112-0082.model)
- [05_refine_resnet152_l2softmax_rgb112-0090.model](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/05_refine_resnet152_l2softmax_rgb112-0090.model)

#### FalsePositive predictor for postprocessing

- [booster_for_val_nms030_tta7_5models_hard_prob.pkl](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/booster_for_val_nms030_tta7_5models_hard_prob.pkl)

### Generate object detection results

Run docker image and run `test_detector.sh`

```
$ ./run.sh
# bash test_detector.sh
... wait many hours  ## takes about 5 hours on a GCP server with V100x8
```

Generates 2 prediction results

- `models/test_060000.pth`
- `models/test_100000.pth`

If you uncommented validation secition of `test_detector.sh`, it generates more 2 prediction results.
This results is needed when you want to reproduce FalsePositive detection model.

- `models/val_060000.pth`
- `models/val_100000.pth`


You can also download the generated object detection results from here.
- [test_060000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/test_060000.pth)
- [test_100000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/test_100000.pth)
- [val_060000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/val_060000.pth)
- [val_100000.pth](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/releases/download/0.0.1/val_100000.pth)


### Generate results with single detector and single model

```
$ bash scripts/auto_generate_per_model_results.sh
```

Generates 10 results

`test_detector_060000_tta7_01_efficientnet_b4`
`test_detector_060000_tta7_02_resnet152`
`test_detector_060000_tta7_03_seresnext101`
`test_detector_060000_tta7_04_seresnext101`
`test_detector_060000_tta7_05_resnet152`
`test_detector_100000_tta7_01_efficientnet_b4`
`test_detector_100000_tta7_02_resnet152`
`test_detector_100000_tta7_03_seresnext101`
`test_detector_100000_tta7_04_seresnext101`
`test_detector_100000_tta7_05_resnet152`


### Ensemble generated results

```
$ python scripts/results_fusion_test_060000.py
$ python scripts/results_fusion_test_100000.py
```

Generates 2 ensemble results

`test_detector_060000_tta7_5models_hard_prob`
`test_detector_100000_tta7_5models_hard_prob`


### NMS with 2 ensemble results

```
$ python scripts/results_nms_test.py
```

Generates nms results
`test_nms030_tta7_5models_hard_prob`

### Postprocessing (FalsePositive predictor)

Remove FalsePositive bbox using LightGBM FalsePositive predictor and
generate final submission csv.


```
$ python scripts/remove_false_positive_and_gen_csv.py
```

Generates `final_submission.csv`


## How to make predictions on a new test set

### Preprocess images

Place images `input/test_images` with jpeg format.

```
$ ls input/test_images
kuzushiji_sample_01.jpg  kuzushiji_sample_02.jpg
```

Replace `input/test_images.list` include your image filename without extension

```
$ cat input/test_images.list
kuzushiji_sample_01
kuzushiji_sample_02
```

```
$ python scripts/denoising_and_bens_preprocessing.py test
./input/denoised_test/kuzushiji_sample_01.png
./input/denoised_test/kuzushiji_sample_02.png
```

### Generate detection results

Remove if object detection results already exists.
```
$ rm models/test_060000.pth
$ rm models/test_100000.pth
```

Generate character detection results
```
$ bash test_detector.sh
```

Generate recognition results

If FileExistsError occurs, delete the target directory and re-execute

```
$ bash scripts/auto_generate_per_model_results.sh
$ python scripts/results_fusion_test_060000.py
$ python scripts/results_fusion_test_100000.py
$ python scripts/results_nms_test.py
$ python scripts/remove_false_positive_and_gen_csv.py
```

### Visualize recognition results

```
$ python scripts/plot_recognition_results.py test_nms030_tta7_5models_hard_prob input/test_images
saved: test_nms030_tta7_5models_hard_prob/kuzushiji_sample_01_with_results.jpg
saved: test_nms030_tta7_5models_hard_prob/kuzushiji_sample_02_with_results.jpg
```

Images containing recognition results are saved under `test_nms030_tta7_5models_hard_prob` directory.


## Reproduce models

Reproducing process of the three models of detection, classification, FalsePositive predictor.


### Reproduce detection model

Detection model use Faster R-CNN with:
- ResNet101 backbone
- Multi-scale train&test
- data augmentation (brightness, contrast, saturation, hue, random grayscale)
- no vertical and horizontal flip

use customized [maskrcnn_benchmark](maskrcnn_benchmark/)

Training Detection model with all train images. And validation with public Leaderboard score.

See config for details. [e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc.yaml](https://github.com/knjcode/kaggle-kuzushiji-recognition-2019/blob/master/configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc.yaml)


Use docker, to reproduce character detection model.

```
$ bash build.sh  ### build docker image
$ bash run.sh  ### run docker container
# cd /work
# bash train_detector.sh
... wait a few hours
```

trained model is saved in the `kuzushiji_recognition_R101_C4` directory.

To generate detection results, use `test_detector.sh` inside docker container.

```
$ ./run.sh
# bash test_detector.sh
... wait a few hours
```


### Reproduce Character classification model

Use scripts under `train_scripts` directory

```
$ bash train_scripts/01_efficientnet_b4_val15779_l2softmax_mixup_re_normalize_gray190.sh
...
```

For details, see [train_scripts/README.md](train_scripts/README.md)


### Reproduce FalsePositive predictor

Train FalsePositive predictor using results of validation data (`val_nms030_tta7_5models_hard_prob`)

#### generate validation resutls

Fix `auto_generate_per_model_results.sh` for validation (uncomment 6th line and comment out 7th line).

```
$ bash scripts/auto_generate_per_model_results.sh
```

Generates 10 validation results

`val_detector_060000_tta7_01_efficientnet_b4`
`val_detector_060000_tta7_02_resnet152`
`val_detector_060000_tta7_03_seresnext101`
`val_detector_060000_tta7_04_seresnext101`
`val_detector_060000_tta7_05_resnet152`
`val_detector_100000_tta7_01_efficientnet_b4`
`val_detector_100000_tta7_02_resnet152`
`val_detector_100000_tta7_03_seresnext101`
`val_detector_100000_tta7_04_seresnext101`
`val_detector_100000_tta7_05_resnet152`


Ensemble generated results

```
$ python scripts/results_fusion_val_060000.py
$ python scripts/results_fusion_val_100000.py
```

Generates 2 ensemble results

`val_detector_060000_tta7_5models_hard_prob`
`val_detector_100000_tta7_5models_hard_prob`


nms with 2 ensemble results

```
$ python scripts/results_nms_val.py
```

generates nms results

`val_nms030_tta7_5models_hard_prob`


#### Hypter parameter search with optuna

Search hyper parameter with validation results.

```
$ python scripts/optuna_search_for_false_positive_detector.py
... wait a few hours
{'lambda_l1': 0.002050689306354841, 'lambda_l2': 0.49425078611198464, 'num_leaves': 203, 'feature_fraction': 0.8606773005600517, 'bagging_fraction': 0.9526576962122715, 'bagging_freq': 2, 'min_child_samples': 66}
```

and generates extracted feature data

`val_nms030_tta7_5models_hard_prob.feather`

#### Train LightGBM model 5fold cv using searched hypter parameters

Train LightGBM FalsePositive predictor reuse generated features at hyper parameter searching.

```
$ python scripts/gen_false_positive_detector.py`
...
saved: models/booster_for_val_nms030_tta7_5models_hard_prob.pkl
```


## License

MIT

