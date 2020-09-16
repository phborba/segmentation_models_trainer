# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-16
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
import json
from absl import flags, app
from segmentation_models_trainer.experiment_builder.experiment import Experiment

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
FLAGS = flags.FLAGS

def main(argv):
    flags.mark_flag_as_required('pipeline_config_path')
    with open(FLAGS.pipeline_config_path) as json_file:
        data = json.load(json_file)
    experiment = Experiment.from_dict(data)
    experiment.train()

if __name__ == '__main__':
  app.run(main)