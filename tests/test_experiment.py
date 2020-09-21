# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-14
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Philipe Borba - Cartographic Engineer @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import unittest
import json
from segmentation_models_trainer.experiment_builder.experiment import Experiment

class Test_TestExperiment(unittest.TestCase):
    experiment = Experiment.from_dict(
        {
            'name' : "test",
            'epochs' : 2,
            'experiment_data_path' : "/data/test",
            'checkpoint_frequency' : 10,
            'warmup_epochs' : 2,
            'use_multiple_gpus' : False,
            'hyperparameters' : {
                'batch_size' : 16,
                'optimizer' : {
                    'name' : "Adam",
                    'config' : {
                        'learning_rate' : 0.01
                    }
                }
            },
            'train_dataset' : json.loads('{"name": "train_ds", "file_path": "/data/train_ds.csv", "n_classes": 1, "dataset_size": 1000, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": true, "shuffle": true, "shuffle_buffer_size": 10000, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 4, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}'),

            'test_dataset' : json.loads('{"name": "test_ds", "file_path": "/data/test_ds.csv", "n_classes": 1, "dataset_size": 1000, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": true, "shuffle": true, "shuffle_buffer_size": 10000, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 4, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}'),

            'model' : json.loads('{"description": "test case", "backbone": "resnet18", "architecture": "Unet", "activation": "sigmoid", "use_imagenet_weights": true}'),
            'loss' : json.loads('{"class_name": "bce_dice_loss", "config": {}, "framework": "sm"}'),
            'callbacks' : json.loads('{"items": [{"name": "ReduceLROnPlateau", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}, {"name": "ModelCheckpoint", "config": {"filepath": "/data/teste"}}]}'),
            'metrics' : json.loads('{"items": [{"class_name": "iou_score", "config": {}, "framework": "sm"}, {"class_name": "precision", "config": {}, "framework": "sm"}, {"class_name": "recall", "config": {}, "framework": "sm"}, {"class_name": "f1_score", "config": {}, "framework": "sm"}, {"class_name": "f2_score", "config": {}, "framework": "sm"}, {"class_name": "LogCoshError", "config": {}, "framework": "tf.keras"}, {"class_name": "KLDivergence", "config": {}, "framework": "tf.keras"}, {"class_name": "MeanIoU", "config": {"num_classes": 2}, "framework": "tf.keras"}]}'),
        }
    )
    json_dict = json.loads('{"name": "test", "epochs": 2, "experiment_data_path": "/data/test", "checkpoint_frequency": 10, "warmup_epochs": 2, "use_multiple_gpus": false, "hyperparameters": {"batch_size": 16, "optimizer": {"name": "Adam", "config": {"learning_rate": 0.01}}}, "train_dataset": {"name": "train_ds", "file_path": "/data/train_ds.csv", "n_classes": 1, "dataset_size": 1000, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": true, "shuffle": true, "shuffle_buffer_size": 10000, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 4, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}, "test_dataset": {"name": "test_ds", "file_path": "/data/test_ds.csv", "n_classes": 1, "dataset_size": 1000, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": true, "shuffle": true, "shuffle_buffer_size": 10000, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 4, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}, "model": {"description": "test case", "backbone": "resnet18", "architecture": "Unet", "activation": "sigmoid", "use_imagenet_weights": true}, "loss": {"class_name": "bce_dice_loss", "config": {}, "framework": "sm"}, "callbacks": {"items": [{"name": "ReduceLROnPlateau", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}, {"name": "ModelCheckpoint", "config": {"filepath": "/data/teste"}}]}, "metrics": {"items": [{"class_name": "iou_score", "config": {}, "framework": "sm"}, {"class_name": "precision", "config": {}, "framework": "sm"}, {"class_name": "recall", "config": {}, "framework": "sm"}, {"class_name": "f1_score", "config": {}, "framework": "sm"}, {"class_name": "f2_score", "config": {}, "framework": "sm"}, {"class_name": "LogCoshError", "config": {}, "framework": "tf.keras"}, {"class_name": "KLDivergence", "config": {}, "framework": "tf.keras"}, {"class_name": "MeanIoU", "config": {"num_classes": 2}, "framework": "tf.keras"}]}}')

    def test_create_instance(self):
        """[summary]
        Tests instance creation
        """          
        self.assertEqual(
            self.experiment.name, 'test'
        )
        self.assertEqual(
            self.experiment.epochs, 2
        )
        self.assertEqual(
            self.experiment.experiment_data_path, '/data/test'
        )
        self.assertEqual(
            self.experiment.checkpoint_frequency, 10
        )
        self.assertEqual(
            self.experiment.warmup_epochs, 2
        )
        self.assertEqual(
            self.experiment.use_multiple_gpus, False
        )
    
    def test_export_instance(self):
        self.assertEqual(
            self.experiment.to_dict(),
            self.json_dict
        )
    
    def test_import_instance(self):
        new_experiment = Experiment.from_dict(
            self.json_dict
        )
        self.assertEqual(
            self.experiment,
            new_experiment
        )