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
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin
from segmentation_models_trainer.model_builder.segmentation_model import SegmentationModel
from segmentation_models_trainer.hyperparameter_builder.hyperparameters import Hyperparameters
from segmentation_models_trainer.dataset_loader.dataset import Dataset

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

    def train(self):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        #using XLA
        tf.config.optimizer.set_jit(True)
        strategy = tf.distribute.MirroredStrategy()

        self.create_data_folders()

        BATCH_SIZE = self.hyperparameters.batch_size * strategy.num_replicas_in_sync \
            if self.use_multiple_gpus else self.hyperparameters.batch_size
        n_classes = self.train_dataset.n_classes
        input_shape = self.train_dataset.get_img_input_shape()
        
        train_ds = self.train_dataset.get_tf_dataset(BATCH_SIZE)
        test_ds = self.test_dataset.get_tf_dataset(BATCH_SIZE)

        training_steps_per_epoch = int( np.ceil(self.train_dataset.dataset_size / BATCH_SIZE) )
        test_steps_per_epoch = int( np.ceil(self.test_dataset.dataset_size / BATCH_SIZE) )

        def train_model(epochs, save_weights_path, encoder_freeze, load_weights=None):
            with strategy.scope():
                model = self.model.get_model(
                    n_classes,
                    encoder_freeze=True,
                    input_shape=input_shape
                )
                opt = self.hyperparameters.optimizer.tf_object
                #TODO metrics and loss fields into compile
                model.compile(
                    opt,
                    loss=sm.losses.bce_jaccard_loss,
                    metrics=[
                        'accuracy',
                        'binary_crossentropy',
                        sm.metrics.iou_score,
                        sm.metrics.precision,
                        sm.metrics.recall,
                        sm.metrics.f1_score,
                        sm.metrics.f2_score
                    ]
                )
            model.fit(
                train_ds,
                batch_size=BATCH_SIZE,
                steps_per_epoch=training_steps_per_epoch,
                epochs=epochs,
                validation_data=test_ds,
                validation_steps=test_steps_per_epoch,
                callbacks=[]
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
            model = train_model(
                epochs=self.warmup_epochs,
                save_weights_path=warmup_path,
                encoder_freeze=True
            )
        final_save_path = os.path.join(
            self.SAVE_PATH,
            'experiment_{name}_{epochs}_epochs.h5'.format(
                name=self.name,
                epochs=self.epochs
            )
        )
        model = train_model(
            epochs=self.warmup_epochs,
            save_weights_path=warmup_path,
            encoder_freeze=False,
            load_weights=warmup_path if self.warmup_epochs > 0 else None
        )
        
        model.save(final_save_path)
    
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

    @staticmethod
    def test_and_create_folder(path):
        if os.path.exists(path):
            return path
        os.makedirs(path, exist_ok=True)
        return path
        


@dataclass
class Callbacks(JsonSchemaMixin):
    name: str
    keras_callback: str

@dataclass
class Metric(JsonSchemaMixin):
    name: str

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

            'model' : json.loads('{"description": "test case", "backbone": "resnet18", "architecture": "Unet", "activation": "sigmoid", "use_imagenet_weights": true}')
        }
    )

    print(experiment.to_json())