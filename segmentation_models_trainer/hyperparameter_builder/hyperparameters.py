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

from segmentation_models_trainer.hyperparameter_builder.optimizer import Optimizer

from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Hyperparameters(JsonSchemaMixin):
    batch_size: int
    optimizer: Optimizer

if __name__ == '__main__':
    x = Optimizer(name='Adam',config={'learning_rate':0.01})
    y = Hyperparameters(batch_size=16, optimizer=x)
    print(y.to_json())