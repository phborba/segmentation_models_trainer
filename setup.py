#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'segmentation_models_trainer'
DESCRIPTION = 'Image segmentation models training of popular architectures.'
URL = 'https://github.com/phborba/segmentation_models_trainer'
EMAIL = 'philipeborba@gmail.com'
AUTHOR = 'Philipe Borba'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1'
LONG_DESCRIPTION = """"
Framework to train semantic segmentation models on TensorFlow using json files as input, as follows:

```
{
    "name": "test",
    "epochs": 4,
    "experiment_data_path": "/data/test",
    "checkpoint_frequency": 10,
    "warmup_epochs": 2,
    "use_multiple_gpus": false,
    "hyperparameters": {
        "batch_size": 16,
        "optimizer": {
            "name": "Adam",
            "config": {
                "learning_rate": 0.0001
            }
        }
    },
    "train_dataset": {
        "name": "train_ds",
        "file_path": "/data/train_ds.csv",
        "n_classes": 1,
        "augmentation_list": [
            {
                "name": "random_crop",
                "parameters": {
                    "crop_width": 256,
                    "crop_height": 256
                }
            },
            {
                "name": "per_image_standardization",
                "parameters": {}
            }
        ],
        "cache": true,
        "shuffle": true,
        "shuffle_buffer_size": 10000,
        "shuffle_csv": true,
        "ignore_errors": true,
        "num_paralel_reads": 4,
        "img_dtype": "float32",
        "img_format": "png",
        "img_width": 512,
        "img_length": 512,
        "use_ds_width_len": false,
        "autotune": -1,
        "distributed_training": false
    },
    "test_dataset": {
        "name": "test_ds",
        "file_path": "/data/test_ds.csv",
        "n_classes": 1,
        "augmentation_list": [
            {
                "name": "random_crop",
                "parameters": {
                    "crop_width": 256,
                    "crop_height": 256
                }
            },
            {
                "name": "random_flip_left_right",
                "parameters": {}
            },
            {
                "name": "random_flip_up_down",
                "parameters": {}
            },
            {
                "name": "random_brightness",
                "parameters": {
                    "max_delta": 0.1
                }
            },
            {
                "name": "random_contrast",
                "parameters": {
                    "lower": 0.5,
                    "upper": 1.5
                }
            },
            {
                "name": "random_saturation",
                "parameters": {
                    "lower": 0.5,
                    "upper": 1.5
                }
            },
            {
                "name": "random_hue",
                "parameters": {
                    "max_delta": 0.01
                }
            },
            {
                "name": "per_image_standardization",
                "parameters": {}
            }
        ],
        "cache": true,
        "shuffle": true,
        "shuffle_buffer_size": 10000,
        "shuffle_csv": true,
        "ignore_errors": true,
        "num_paralel_reads": 4,
        "img_dtype": "float32",
        "img_format": "png",
        "img_width": 512,
        "img_length": 512,
        "use_ds_width_len": false,
        "autotune": -1,
        "distributed_training": false
    },
    "model": {
        "description": "test case",
        "backbone": "resnet18",
        "architecture": "Unet",
        "activation": "sigmoid",
        "use_imagenet_weights": true
    },
    "loss": {
        "class_name": "bce_dice_loss",
        "config": {},
        "framework": "sm"
    },
    "callbacks": {
        "items": [
            {
                "name": "TensorBoard",
                "config": {
                    "update_freq": "epoch"
                }
            },
            {
                "name": "BackupAndRestore",
                "config": {}
            },
            {
                "name": "ReduceLROnPlateau",
                "config": {
                    "monitor": "val_loss",
                    "factor": 0.2,
                    "patience": 5,
                    "min_lr": 0.00000000001
                }
            },
            {
                "name": "ModelCheckpoint",
                "config": {
                    "monitor": "iou_score",
                    "save_best_only": false,
                    "save_weights_only": false,
                    "verbose":1
                }
            },
            {
                "name": "ImageHistory",
                "config": {
                    "draw_interval": 1,
                    "page_size": 10
                }
            }
        ]
    },
    "metrics": {
        "items": [
            {
                "class_name": "iou_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "precision",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "recall",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "f1_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "f2_score",
                "config": {},
                "framework": "sm"
            },
            {
                "class_name": "MeanIoU",
                "config": {
                    "num_classes": 2
                },
                "framework": "tf.keras"
            }
        ]
    }
}
```


Training usage:

```
python train.py --pipeline_config_path=my_experiment.json

```
"""

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        REQUIRED = []
        DEP_LINKS = []
        for i in f.read().split('\n'):
            if 'git+' in i:
                DEP_LINKS.append(i)
            else:
                REQUIRED.append(i)

except:
    REQUIRED = []
    DEP_LINKS = []

# What packages are optional?
EXTRAS = {
    'tests': ['pytest', 'scikit-image'],
}

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print(s)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', 'docs', 'images', 'examples')),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEP_LINKS,
    include_package_data=True,
    license='GPL',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    keywords='tensorflow keras semantic-segmentation deep learning',
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)