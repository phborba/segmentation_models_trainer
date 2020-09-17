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
class Callback(JsonSchemaMixin):
    name: int
    parameters: dict
    def __post_init__(self):
        pass

    @staticmethod
    def validate_callback_name(name):
        if name not in [
            i for i in dir(tf.keras.callbacks) + dir(tf.keras.callbacks.experimental)\
                if '__' not in i
            ]:
            raise ValueError("Callback not implemented")

    def get_callback(self):
        pass



if __name__ == '__main__':
    x = Callback(
        name='ReduceLROnPlateau',
        parameters={
            monitor='val_loss',
            factor=0.1,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        }
    )
    print(x.to_json())