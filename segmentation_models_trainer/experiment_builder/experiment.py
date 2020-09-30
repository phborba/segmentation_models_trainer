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
import tensorflow as tf
import segmentation_models as sm
import os
import numpy as np
import importlib
from typing import Any, List
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin
from segmentation_models_trainer.model_builder.segmentation_model import SegmentationModel
from segmentation_models_trainer.hyperparameter_builder.hyperparameters import Hyperparameters
from segmentation_models_trainer.dataset_loader.dataset import Dataset
from segmentation_models_trainer.callbacks_loader.callback import Callback, CallbackList
from segmentation_models_trainer.experiment_builder.loss import Loss
from segmentation_models_trainer.experiment_builder.metric import Metric, MetricList

@dataclass
class Experiment(JsonSchemaMixin):
    #TODO add loss field
    #TODO add metrics
    name: str
    epochs: int
    experiment_data_path: str
    checkpoint_frequency: int
    warmup_epochs: int
    use_multiple_gpus: bool
    hyperparameters: Hyperparameters
    train_dataset: Dataset
    test_dataset: Dataset
    model: SegmentationModel
    loss: Loss
    callbacks: CallbackList
    metrics: MetricList

    def train(self):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        #using XLA
        tf.config.optimizer.set_jit(True)
        strategy = tf.distribute.MirroredStrategy()

        self.create_data_folders()

        self.BATCH_SIZE = self.hyperparameters.batch_size * strategy.num_replicas_in_sync \
            if self.use_multiple_gpus else self.hyperparameters.batch_size
        n_classes = self.train_dataset.n_classes
        input_shape = self.train_dataset.get_img_input_shape()
        
        train_ds = self.train_dataset.get_tf_dataset(self.BATCH_SIZE)
        test_ds = self.test_dataset.get_tf_dataset(self.BATCH_SIZE)

        self.training_steps_per_epoch = int( np.ceil(self.train_dataset.dataset_size / self.BATCH_SIZE) )
        self.test_steps_per_epoch = int( np.ceil(self.test_dataset.dataset_size / self.BATCH_SIZE) )

        def train_model(epochs, save_weights_path, encoder_freeze, callback_list, load_weights=None):
            with strategy.scope():
                model = self.model.get_model(
                    n_classes,
                    encoder_freeze=encoder_freeze,
                    input_shape=input_shape
                )
                opt = self.hyperparameters.optimizer.tf_object
                metric_list = self.metrics.get_tf_objects()
                model.compile(
                    opt,
                    loss=self.loss.loss_obj,
                    metrics=metric_list
                )
                if load_weights is not None:
                    model.load_weights(load_weights)
            model.fit(
                train_ds,
                batch_size=self.BATCH_SIZE,
                steps_per_epoch=self.training_steps_per_epoch,
                epochs=epochs,
                validation_data=test_ds,
                validation_steps=self.test_steps_per_epoch,
                callbacks=callback_list
            )
            model.save_weights(
                save_weights_path
            )
            return model

        if self.warmup_epochs > 0:
            warmup_path = os.path.join(
                self.SAVE_PATH,
                'warmup_experiment_{name}.h5'.format(name=self.name)
            )
            callback_list = self.get_initialized_callbacks(
                epochs=self.warmup_epochs,
                data_ds=train_ds,
                warmup=True
            )
            model = train_model(
                epochs=self.warmup_epochs,
                save_weights_path=warmup_path,
                encoder_freeze=True,
                callback_list=callback_list
            )
        final_save_path = os.path.join(
            self.SAVE_PATH,
            'experiment_{name}_{epochs}_epochs.h5'.format(
                name=self.name,
                epochs=self.epochs
            )
        )
        callback_list = self.get_initialized_callbacks(
            epochs=self.warmup_epochs,
            data_ds=train_ds,
            warmup=False
        )
        model = train_model(
            epochs=self.epochs,
            save_weights_path=final_save_path,
            encoder_freeze=False,
            callback_list=callback_list,
            load_weights=warmup_path if self.warmup_epochs > 0 else None
        )
    
    def create_data_folders(self):
        DATA_DIR = self.test_and_create_folder(self.experiment_data_path)
        self.TRAIN_CACHE = self.test_and_create_folder(
            os.path.join(DATA_DIR, 'cache', 'train_cache')
        )
        self.TEST_CACHE = self.test_and_create_folder(
            os.path.join(DATA_DIR, 'cache', 'train_cache')
        )
        self.LOG_PATH = self.test_and_create_folder(
            os.path.join(DATA_DIR,'logs', 'scalars')
        )
        self.CHECKPOINT_PATH = self.test_and_create_folder(
            os.path.join(DATA_DIR, 'logs', 'checkpoints')
        )
        self.SAVE_PATH = self.test_and_create_folder(
            os.path.join(DATA_DIR, 'saved_models')
        )
        self.REPORT_DIR = self.test_and_create_folder(
            os.path.join(DATA_DIR, 'report_img')
        )
    
    def get_initialized_callbacks(self, epochs, data_ds, warmup=False):
        tf_callback_list = []
        for callback in self.callbacks.items:
            if callback.name == 'BackupAndRestore':
                callback.config.update({'backup_dir':self.CHECKPOINT_PATH})
            elif callback.name == 'ImageHistory':
                callback.config.update(
                    {
                        'dataset' : data_ds,
                        'tensorboard_dir' : self.LOG_PATH,
                        'n_epochs' : epochs,
                        'batch_size' : self.BATCH_SIZE,
                        'report_dir' : self.REPORT_DIR
                    }
                )
            elif callback.name == 'ModelCheckpoint':
                callback.config.update(
                    {
                        'filepath': os.path.join(
                            self.CHECKPOINT_PATH,
                            "model-{epoch:04d}.ckpt" if not warmup else "warmup_model-{epoch:04d}.ckpt"
                        ),
                        'save_freq': self.checkpoint_frequency * self.training_steps_per_epoch
                    }
                )
            tf_callback_list.append(
                callback.get_callback()
            )
        return tf_callback_list

    @staticmethod
    def test_and_create_folder(path):
        if os.path.exists(path):
            return path
        os.makedirs(path, exist_ok=True)
        return path

if __name__ == "__main__":
    import json
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
            'callbacks' : json.loads('{"items": [{"name": "ReduceLROnPlateau", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}, {"name": "ModelCheckpoint", "config": {"filepath": "/data/teste/checkpoint.hdf5"}}]}'),
            'metrics' : json.loads('{"items": [{"class_name": "iou_score", "config": {}, "framework": "sm"}, {"class_name": "precision", "config": {}, "framework": "sm"}, {"class_name": "recall", "config": {}, "framework": "sm"}, {"class_name": "f1_score", "config": {}, "framework": "sm"}, {"class_name": "f2_score", "config": {}, "framework": "sm"}, {"class_name": "LogCoshError", "config": {}, "framework": "tf.keras"}, {"class_name": "KLDivergence", "config": {}, "framework": "tf.keras"}, {"class_name": "MeanIoU", "config": {"num_classes": 2}, "framework": "tf.keras"}]}'),
        }
    )

    print(experiment.to_json())