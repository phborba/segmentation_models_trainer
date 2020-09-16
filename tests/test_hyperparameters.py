# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-14
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Philipe Borba - Cartographic Engineer 
                                                            @ Brazilian Army
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
from segmentation_models_trainer.hyperparameter_builder.hyperparameters import \
    Hyperparameters
from segmentation_models_trainer.hyperparameter_builder.optimizer import \
    Optimizer

class Test_TestHyperparameters(unittest.TestCase):
    opt = Optimizer(name='Adam',config={'learning_rate':0.01})
    hp = Hyperparameters(batch_size=16, optimizer=opt)
    opt_json_text = '''{"name": "Adam", "config": {"learning_rate": 0.01}}'''
    hp_json_text = '{"batch_size": 16, "optimizer": {"name": "Adam", "config": {"learning_rate": 0.01}}}'

    def test_create_instance(self):
        self.assertEqual(
            self.hp.batch_size, 16
        )
        self.assertEqual(
            self.hp.optimizer, self.opt
        )
        self.assertEqual(
            self.opt.to_json(), self.opt_json_text
        )
        self.assertEqual(
            self.hp.to_json(), self.hp_json_text
        )
