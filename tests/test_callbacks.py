# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-21
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
import segmentation_models as sm
from segmentation_models_trainer.callbacks_loader.callback import Callback
import numpy as np
import tensorflow as tf


class Test_TestCallbacks(unittest.TestCase):
    
    callback = Callback(
        name='ReduceLROnPlateau',
        config= {
            'monitor' : 'val_loss',
            'factor' : 0.2,
            'patience' : 5,
            'min_lr' : 0.001
        }
    )
    json_dict = json.loads('{"name": "ReduceLROnPlateau", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}')

    def test_create_instance(self):
        """[summary]
        Tests instance creation
        """          
        self.assertEqual(
            self.callback.name, 'ReduceLROnPlateau'
        )

    def test_export_instance(self):
        self.assertEqual(
            self.callback.to_dict(),
            self.json_dict
        )
    
    def test_import_instance(self):
        new_callback = Callback.from_dict(
            self.json_dict
        )
        self.assertEqual(
            self.callback,
            new_callback
        )
    
class Test_TestCustomCallbacks(Test_TestCallbacks):
    callback = Callback(
        name='ImageHistory',
        config= {
            'monitor' : 'val_loss',
            'factor' : 0.2,
            'patience' : 5,
            'min_lr' : 0.001
        }
    )
    json_dict = json.loads('{"name": "ImageHistory", "config": {"monitor": "val_loss", "factor": 0.2, "patience": 5, "min_lr": 0.001}}')
    def test_create_instance(self):
        """[summary]
        Tests instance creation
        """          
        self.assertEqual(
            self.callback.name, 'ImageHistory'
        )

if __name__ == '__main__':
    x = Test_TestCustomCallbacks()
    x