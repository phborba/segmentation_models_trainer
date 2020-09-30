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
from segmentation_models_trainer.dataset_loader.dataset import Dataset, ImageAugumentation


class Test_TestDataset(unittest.TestCase):
    aug_list = [
        ImageAugumentation.from_dict(
            {
                'name' : 'random_crop',
                'parameters' : {
                    'crop_width' : 256,
                    'crop_height' : 256
                }
            }
        ),
        ImageAugumentation.from_dict(
            {
                'name' : 'per_image_standardization',
                'parameters' : {
                }
            }
        )
    ]
    dataset = Dataset(
        name='test',
        file_path='/data/test',
        n_classes=1,
        dataset_size=1000,
        augmentation_list=aug_list
    )
    json_dict = json.loads('{"name": "test", "n_classes": 1, "dataset_size": 1000, "augmentation_list": [{"name": "random_crop", "parameters": {"crop_width": 256, "crop_height": 256}}, {"name": "per_image_standardization", "parameters": {}}], "file_path": "/data/test", "base_path": "", "cache": true, "shuffle": true, "shuffle_buffer_size": 10000, "shuffle_csv": true, "ignore_errors": true, "num_paralel_reads": 4, "img_dtype": "float32", "img_format": "png", "img_width": 512, "img_length": 512, "img_bands": 3, "mask_bands": 1, "use_ds_width_len": false, "autotune": -1, "distributed_training": false}')

    def test_create_instance(self):
        """[summary]
        Tests instance creation
        """          
        self.assertEqual(
            self.dataset.name, 'test'
        )
        self.assertEqual(
            self.dataset.file_path, '/data/test'
        )
        self.assertEqual(
            self.dataset.n_classes, 1
        )
    
    def test_export_instance(self):
        self.assertEqual(
            self.dataset.to_dict(),
            self.json_dict
        )
    
    def test_import_instance(self):
        new_dataset = Dataset.from_dict(
            self.json_dict
        )
        self.assertEqual(
            self.dataset,
            new_dataset
        )