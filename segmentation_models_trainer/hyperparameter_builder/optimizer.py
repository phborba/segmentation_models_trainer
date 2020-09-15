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
import tensorflow as tf
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Optimizer(JsonSchemaMixin):
    name: str
    config: dict

    def __post_init__(self):
        self.tf_object = tf.keras.optimizers.deserialize(
            {
                'class_name' : self.name,
                'config' : self.config
            }
        )


if __name__ == '__main__':
    x = Optimizer(name='Adam',config={'learning_rate':0.01})
    obj = x.tf_object
    print(obj.get_config())
    print(x.to_json())