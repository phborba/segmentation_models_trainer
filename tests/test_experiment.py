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
from segmentation_models_trainer.core.experiment_builder.experiment import Experiment

class Test_TestExperiment(unittest.TestCase):

    def test_create_instance(self):
        experiment = Experiment(
            name='test',
            epochs=2,
            log_path='/data/test',
            checkpoint_frequency=10,
            warmup_epochs=2,
            batch_size=16,
            use_multiple_gpus=False
        )
        self.assertEqual(
            experiment.name, 'test'
        )
        self.assertEqual(
            experiment.epochs, 2
        )