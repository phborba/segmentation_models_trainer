# segmentation_models_trainer

![Python application](https://github.com/phborba/segmentation_models_trainer/workflows/Python%20application/badge.svg)
[![maintainer](https://img.shields.io/badge/maintainer-phborba-blue.svg)](https://github.com/phborba)
[![DOI](https://zenodo.org/badge/294972255.svg)](https://zenodo.org/badge/latestdoi/294972255)

Framework to train semantic segmentation models on TensorFlow using json files as input, as follows:


```
{
    "name": "test",
    "epochs": 4,
    "experiment_data_path": "/data/test",
    "checkpoint_frequency": 10,
    "warmup_epochs": 2,
    "use_multiple_gpus": false,
    "hyperparameters": {
        "batch_size": 16,
        "optimizer": {
            "name": "Adam",
            "config": {
                "learning_rate": 0.0001
            }
        }
    },
    "train_dataset": {
        "name": "train_ds",
        "file_path": "/data/train_ds.csv",
        "n_classes": 1,
        "dataset_size": 1000,
        "augmentation_list": [
            {
                "name": "random_crop",
                "parameters": {
                    "crop_width": 256,
                    "crop_height": 256
                }
            },
            {
                "name": "per_image_standardization",
                "parameters": {}
            }
        ],
        "cache": true,
        "shuffle": true,
        "shuffle_buffer_size": 10000,
        "shuffle_csv": true,
        "ignore_errors": true,
        "num_paralel_reads": 4,
        "img_dtype": "float32",
        "img_format": "png",
        "img_width": 512,
        "img_length": 512,
        "use_ds_width_len": false,
        "autotune": -1,
        "distributed_training": false
    },
    "test_dataset": {
        "name": "test_ds",
        "file_path": "/data/test_ds.csv",
        "n_classes": 1,
        "dataset_size": 200,
        "augmentation_list": [
            {
                "name": "random_crop",
                "parameters": {
                    "crop_width": 256,
                    "crop_height": 256
                }
            },
            {
                "name": "random_flip_left_right",
                "parameters": {}
            },
            {
                "name": "random_flip_up_down",
                "parameters": {}
            },
            {
                "name": "random_brightness",
                "parameters": {
                    "max_delta": 0.1
                }
            },
            {
                "name": "random_contrast",
                "parameters": {
                    "lower": 0.5,
                    "upper": 1.5
                }
            },
            {
                "name": "random_saturation",
                "parameters": {
                    "lower": 0.5,
                    "upper": 1.5
                }
            },
            {
                "name": "random_hue",
                "parameters": {
                    "max_delta": 0.01
                }
            },
            {
                "name": "per_image_standardization",
                "parameters": {}
            }
        ],
        "cache": true,
        "shuffle": true,
        "shuffle_buffer_size": 10000,
        "shuffle_csv": true,
        "ignore_errors": true,
        "num_paralel_reads": 4,
        "img_dtype": "float32",
        "img_format": "png",
        "img_width": 512,
        "img_length": 512,
        "use_ds_width_len": false,
        "autotune": -1,
        "distributed_training": false
    },
    "model": {
        "description": "test case",
        "backbone": "resnet18",
        "architecture": "Unet",
        "activation": "sigmoid",
        "use_imagenet_weights": true
    },
    "loss": {
        "class_name": "bce_dice_loss",
        "config": {},
        "framework": "sm"
    },
    "callbacks": {
        "items": [
            {
                "name": "TensorBoard",
                "config": {
                    "update_freq": "epoch"
                }
            },
            {
                "name": "BackupAndRestore",
                "config": {}
            },
            {
                "name": "ReduceLROnPlateau",
                "config": {
                    "monitor": "val_loss",
                    "factor": 0.2,
                    "patience": 5,
                    "min_lr": 0.00000000001
                }
            },
            {
                "name": "ModelCheckpoint",
                "config": {
                    "monitor": "iou_score",
                    "save_best_only": false,
                    "save_weights_only": false,
                    "verbose":1
                }
            },
            {
                "name": "ImageHistory",
                "config": {
                    "draw_interval": 1,
                    "page_size": 10
                }
            }
        ]
    },
    "metrics": {
        "items": [
            {
                "class_name": "iou_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "precision",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "recall",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "f1_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "f2_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "MeanIoU",
                "config": {
                    "num_classes": 2
                },
                "framework": "tf.keras"
            }
        ]
    }
}
```


Training usage:

```
python train.py --pipeline_config_path=my_experiment.json

```

Citing:
```
@software{philipe_borba_2020_4060390,
  author       = {Philipe Borba},
  title        = {phborba/segmentation\_models\_trainer: First Release},
  month        = sep,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.4060390},
  url          = {https://doi.org/10.5281/zenodo.4060390}
}
```
