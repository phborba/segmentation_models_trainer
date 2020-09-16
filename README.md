# segmentation_models_trainer

![Python application](https://github.com/phborba/segmentation_models_trainer/workflows/Python%20application/badge.svg)
[![maintainer](https://img.shields.io/badge/maintainer-phborba-blue.svg)](https://github.com/phborba)

Framework to train semantic segmentation models on TensorFlow using json files as input, as follows:


```
{
    "name": "Example",
    "epochs": 100,
    "experiment_data_path": "/data/test",
    "checkpoint_frequency": 1,
    "warmup_epochs": 2,
    "use_multiple_gpus": false,
    "hyperparameters": {
        "batch_size": 16,
        "optimizer": {
            "name": "Adam",
            "config": {
                "learning_rate": 0.01
            }
        }
    },
    "train_dataset": {
        "name": "train_ds",
        "file_path": "/data/dataset_test.csv",
        "n_classes": 1,
        "dataset_size": 1000,
        "dataset_size": 60000,
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
        "img_width": 256,
        "img_length": 256,
        "use_ds_width_len": false,
        "autotune": -1,
        "distributed_training": false
    },
    "test_dataset": {
        "name": "test_ds",
        "file_path": "/data/dataset_test.csv",
        "n_classes": 1,
        "dataset_size": 1000,
        "dataset_size": 40000,
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
    "model": {
        "description": "test case",
        "backbone": "resnet18",
        "architecture": "Unet",
        "activation": "sigmoid",
        "use_imagenet_weights": true
    }
}
```


Training usage:

```
python train.py --pipeline_config_path=my_experiment.json

```