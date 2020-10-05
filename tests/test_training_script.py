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
import os
import shutil
import subprocess
from segmentation_models_trainer.hyperparameter_builder.optimizer import Optimizer
from tensorflow.keras.optimizers import (
    Adam, SGD, RMSprop
)

def get_file_list(dir_path, extension):
    output_list = []
    for root, dirs, files in os.walk(dir_path):
        output_list += [os.path.join(root,f) for f in files if f.endswith(extension)]
    return sorted(output_list)

def create_csv_file(file_path, image_list, label_list):
    csv_text = 'id,image_path,label_path,rows,columns\n'
    for idx, i in enumerate(image_list):
        csv_text += f"{idx},{i},{label_list[idx]},512,512\n"
    with open(file_path, 'w') as csv_file:
        csv_file.write(csv_text)
    return file_path

class Test_TestTrainingScript(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        image_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'images'),
            '.png'
        )
        label_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'labels'),
            '.png'
        )
        label_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'labels'),
            '.png'
        )
        self.csv_train_ds_file = create_csv_file(
            os.path.join(current_dir, 'testing_data', 'csv_train_ds.csv'),
            image_list[0:5],
            label_list[0:5]
        )
        
        self.csv_test_ds_file = create_csv_file(
            os.path.join(current_dir, 'testing_data', 'csv_test_ds.csv'),
            [
                os.path.join('/', *i.split('/')[-2::]) for i in image_list[5::]
            ],
            [
                os.path.join('/', *i.split('/')[-2::]) for i in label_list[5::]
            ]
        )#different label list procedure to test loading data with data path prefix
        self.experiment_path = os.path.join(
                current_dir,
                'experiment_data'
        )
        if not os.path.exists(self.experiment_path):
            os.makedirs(
                self.experiment_path
            )
        settings_dict = {
            'name' : "test",
            'epochs' : 2,
            'experiment_data_path' : self.experiment_path,
            'checkpoint_frequency' : 1,
            'warmup_epochs' : 1,
            'use_multiple_gpus' : False,
            'hyperparameters' : {
                'batch_size' : 2,
                'optimizer' : {
                    'name' : "Adam",
                    'config' : {
                        'learning_rate' : 0.01
                    }
                }
            },
            'train_dataset' : json.loads(
                '''
                {"name": "train_ds", "file_path": "'''+self.csv_train_ds_file+'''", "n_classes": 1, "dataset_size": 5, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": false, "shuffle": false, "shuffle_buffer_size": 1, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 1, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}
                '''
            ),

            'test_dataset' : json.loads(
                '''{"name": "test_ds", "file_path": "'''+self.csv_test_ds_file+'''", "n_classes": 1, "dataset_size": 5, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "cache": false, "shuffle": false, "shuffle_buffer_size": 1, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 1, "img_dtype": "float32", "img_format": "png", "img_width": 256, "img_length": 256, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}'''
            ),

            'model' : json.loads('{"description": "test case", "backbone": "resnet18", "architecture": "Unet", "activation": "sigmoid", "use_imagenet_weights": true}'),
            'loss' : json.loads('{"class_name": "bce_dice_loss", "config": {}, "framework": "sm"}'),
            'callbacks' : json.loads('{"items": [{"name": "TensorBoard", "config": {"update_freq": "epoch"}}, {"name": "ReduceLROnPlateau", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}, {"name": "ModelCheckpoint", "config": {"monitor": "iou_score", "save_best_only": false, "save_weights_only": false, "verbose": 1}}, {"name": "ImageHistory", "config": {"draw_interval": 1, "page_size": 2}}]}'),
            'metrics' : json.loads('{"items": [{"class_name": "iou_score", "config": {}, "framework": "sm"}, {"class_name": "precision", "config": {}, "framework": "sm"}, {"class_name": "recall", "config": {}, "framework": "sm"}, {"class_name": "f1_score", "config": {}, "framework": "sm"}, {"class_name": "f2_score", "config": {}, "framework": "sm"}, {"class_name": "LogCoshError", "config": {}, "framework": "tf.keras"}, {"class_name": "KLDivergence", "config": {}, "framework": "tf.keras"}, {"class_name": "MeanIoU", "config": {"num_classes": 2}, "framework": "tf.keras"}]}'),
        }
        self.settings_json = os.path.join(
            current_dir, 'testing_data', 'settings.json'
        )
        settings_dict['test_dataset']['base_path'] = os.path.join(
            os.path.dirname(__file__),
            'testing_data',
            'data'
        )
        with open(self.settings_json, 'w') as json_file:
            json_file.write(json.dumps(settings_dict))
        
    def tearDown(self):
        os.remove(self.csv_train_ds_file)
        os.remove(self.csv_test_ds_file)
        os.remove(self.settings_json)
        shutil.rmtree(self.experiment_path)
    
    def test_execute_script(self):
        script_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'segmentation_models_trainer',
            'train.py'

        )
        returnFromProcess = subprocess.run(
            ['python3', script_path,'--pipeline_config_path', self.settings_json]
        )
        self.assertEqual(
            returnFromProcess.returncode,
            0
        )
