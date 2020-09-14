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
from segmentation_models_trainer.hyperparameter_builder.optimizer import Optimizer
from tensorflow.keras.optimizers import (
    Adam, SGD, RMSprop
)

class Test_TestOptimizer(unittest.TestCase):
    tf_object_dict = {
        'Adam' : Adam(learning_rate=0.1),
        'SGD' : SGD(learning_rate=0.1),
        'RMSProp' : RMSprop(learning_rate=0.1)
    }
    optimizer_dict = {
        'Adam' : Optimizer(name='Adam', config={'learning_rate': 0.1}),
        'SGD' : Optimizer(name='SGD', config={'learning_rate': 0.1}),
        'RMSProp' : Optimizer(name='RMSProp', config={'learning_rate': 0.1})
    }

    def test_create_instance(self):
        for name, opt in self.optimizer_dict.items():
            self.assertEqual(
                opt.name, name
            )
            self.assertEqual(
                opt.config, {'learning_rate': 0.1}
            )
            self.assertEqual(
                opt.tf_object.get_config(), self.tf_object_dict[name].get_config()
            )