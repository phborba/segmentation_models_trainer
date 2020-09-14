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

from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Experiment(JsonSchemaMixin):
    name: str
    epochs: int
    log_path: str
    checkpoint_frequency: int
    warmup_epochs: int
    use_multiple_gpus: bool

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
                self.name,
                self.epochs,
                self.log_path,
                self.checkpoint_frequency,
                self.warmup_epochs,
                self.use_multiple_gpus
            ) == (
                other.name,
                other.epochs,
                other.log_path,
                other.checkpoint_frequency,
                other.warmup_epochs,
                other.use_multiple_gpus
            )


@dataclass
class Callbacks(JsonSchemaMixin):
    name: str
    keras_callback: str

if __name__ == "__main__":
    import pprint
    import json
    pp = pprint.PrettyPrinter(indent=4)
    experiment = Experiment(
        name='test',
        epochs=2,
        log_path='/data/test',
        checkpoint_frequency=10,
        warmup_epochs=2,
        use_multiple_gpus=False
    )
    pp.pprint(experiment.json_schema())
    print(json.dumps(experiment.to_dict()))