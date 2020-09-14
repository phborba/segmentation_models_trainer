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
    experiment = Experiment(
        name='test',
        epochs=2,
        log_path='/data/test',
        checkpoint_frequency=10,
        warmup_epochs=2,
        use_multiple_gpus=False
    )
    json_dict = json.loads("""{"name": "test", "epochs": 2, "log_path": "/data/test", "checkpoint_frequency": 10, "warmup_epochs": 2, "use_multiple_gpus": false}""")

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
            self.experiment.log_path, '/data/test'
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