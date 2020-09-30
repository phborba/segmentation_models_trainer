# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-15
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
import os
from dataclasses import dataclass, field
from dataclasses_jsonschema import JsonSchemaMixin
from typing import Any, List
from collections import OrderedDict


IMAGE_DTYPE = {
    'float32' : tf.float32,
    'float16' : tf.float16
}

@tf.function
def get_augmentation(name, image, parameters):
    """
        Returns augmentation function. It has to be implemented
        as if else, so that tensorflow can convert it to faster
        implementation.

    Args:
        name (str): name of the augmentation

    Returns:
        tf function: function to perform augmentation
    """
    if name == 'random_crop':
        return tf.image.random_crop(
            image,
            size=[
                parameters['crop_width'],
                parameters['crop_height'],
                image.shape[-1]
            ]
        )
    elif name == 'random_flip_left_right':
        return tf.image.random_flip_left_right(
            image
        )
    elif 'random_flip_up_down':
        return tf.image.random_flip_up_down(
            image
        )
    elif 'random_brightness':
        return tf.image.random_brightness(
            image,
            parameters['max_delta']
        )
    elif 'random_contrast':
        return tf.image.random_contrast(
            image,
            parameters['lower'],
            parameters['upper']
        )
    elif 'random_saturation':
        return tf.image.random_saturation(
            image,
            parameters['lower'],
            parameters['upper']
        )
    elif 'random_hue':
        return tf.image.random_hue(
            image,
            parameters['max_delta']
        )
    elif 'per_image_standardization':
        return tf.image.per_image_standardization(
            image
        )
    else:
        raise ValueError("Augmentation not implemented.")

@dataclass
class ImageAugumentation(JsonSchemaMixin):
    name: str
    parameters: dict

    def augment_image(self, input_image, input_mask):
        image = tf.concat(
                values=[input_image, input_mask], axis=-1
            ) if self.name in [
                'random_crop',
                'random_flip_left_right',
                'random_flip_up_down'
            ] else input_image
        augmented_image = get_augmentation(
            self.name,
            image,
            self.parameters
        )
        if self.name in [
                'random_crop', 'random_flip_left_right', 'random_flip_up_down'
            ]:
            image = augmented_image[:, :, :-1]
            mask = augmented_image[:, :, -1:]
        else:
            image = augmented_image
            mask = input_mask
        return image, mask

@dataclass
class Dataset(JsonSchemaMixin):
    name: str
    n_classes: int
    dataset_size: int
    augmentation_list: List[ImageAugumentation]
    file_path: str
    base_path: str = ''
    cache: Any = True
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    shuffle_csv: bool = True
    ignore_errors: bool = True
    num_paralel_reads: int = 4
    img_dtype: str = 'float32'
    img_format: str = 'png'
    img_width: int = 512
    img_length: int = 512
    img_bands: int = 3
    mask_bands: int = 1
    use_ds_width_len: bool = False
    autotune: int = tf.data.experimental.AUTOTUNE
    distributed_training: bool = False

    def get_img_input_shape(self):
        return (
            self.img_width,
            self.img_length,
            self.img_bands
        )

    def get_tf_dataset(self, batch_size):
        @tf.function
        def process_csv_entry(entry):
            width = entry['width'] if self.use_ds_width_len else self.img_width
            length = entry['length'] if self.use_ds_width_len else self.img_length
            label = tf.io.read_file(
                entry['label_path'][0] if self.base_path == '' \
                    else tf.strings.join(
                        [
                            self.base_path,
                            entry['label_path'][0]
                        ],
                        separator=''
                    )
            )
            label = decode_img(label, width, length, channels=1)
            # load the raw data from the file as a string
            img = tf.io.read_file(
                entry['label_path'][0] if self.base_path == '' \
                    else tf.strings.join(
                        [
                            self.base_path,
                            entry['label_path'][0]
                        ],
                        separator=''
                    )
            )
            img = decode_img(img, width, length)
            img, label = augment_image(img, label)
            return img, label
        
        
        def decode_img(img, width, length, channels=3):
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_png(img, channels=channels)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, IMAGE_DTYPE[self.img_dtype])
            # resize the image to the desired size.
            return tf.image.resize(img, [width, length])
        
        def prepare_for_training(ds, batch_size):
            if self.cache:
                if isinstance(self.cache, str):
                    ds = ds.cache(self.cache)
                else:
                    ds = ds.cache()
            if self.shuffle:
                ds = ds.shuffle(self.shuffle_buffer_size)
            ds = ds.batch(batch_size)
            # Repeat forever
            ds = ds.repeat()
            # `prefetch` lets the dataset fetch batches in the background while the model
            # is training.
            ds = ds.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE \
                    if self.autotune == -1 else self.autotune
            )
            return ds
        
        def augment_image(img, label):
            for augmentation in self.augmentation_list:
                img, label = augmentation.augment_image(
                    img,
                    label
                )
            return img, label

        ds = tf.data.experimental.make_csv_dataset(
            self.file_path,
            batch_size=batch_size,
            ignore_errors=self.ignore_errors,
            num_parallel_reads=self.num_paralel_reads
        )
        labeled_ds = ds.map(
            process_csv_entry,
            num_parallel_calls=tf.data.experimental.AUTOTUNE if self.autotune == -1 \
                else self.autotune
        )
        prepared_ds = prepare_for_training(
            labeled_ds,
            batch_size
        )
        if not self.distributed_training:
            return prepared_ds
        else:
            strategy = tf.distribute.get_strategy()
            return strategy.experimental_distribute_dataset(prepared_ds)

if __name__ == '__main__':
    aug_list = [
        ImageAugumentation.from_dict(
            {
                'name' : 'random_crop',
                'parameters' : {
                    'crop_width' : 256,
                    'crop_height' : 256
                }
            }
        ),
        ImageAugumentation.from_dict(
            {
                'name' : 'per_image_standardization',
                'parameters' : {
                }
            }
        )
    ]
    y = Dataset(
        name='test',
        file_path='/data/test',
        n_classes=1,
        dataset_size=1000,
        augmentation_list=aug_list
    )
    print(y.to_json())