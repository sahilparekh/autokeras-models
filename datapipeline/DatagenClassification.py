import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import os
from functools import partial

MEAN_RGB = tf.constant([127.0, 127.0, 127.0], shape=[1, 1, 3], dtype=tf.float32)
STDDEV_RGB = tf.constant([128.0, 128.0, 128.0], shape=[1, 1, 3], dtype=tf.float32)

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/class/id':
        tf.io.VarLenFeature(tf.int64),
    'image/class/text':
        tf.io.VarLenFeature(tf.string),
}


class TFClassificationDataGenerator(object):

    def __init__(self,
                 train_files,
                 val_files,
                 num_classes,
                 train_aug=None,
                 val_aug=None,
                 include_bgclass=False,
                 train_size=None,
                 val_size=None,
                 repeatds=True,
                 img_size=None,
                 normalize_image=False):

        assert len(val_files) > 0, 'No validation files found'
        assert len(train_files) > 0, 'No training files found'
        assert train_size > 0, 'Please provide train ds size for ease of use'
        assert val_size > 0, 'Please provide validation ds size for ease of use'

        self.num_classes = num_classes
        self.include_bgclass = include_bgclass
        self.effective_ncls = self.num_classes + 1 if self.include_bgclass else self.num_classes
        self.repeatds = repeatds
        self.val_raw_ds = tf.data.TFRecordDataset(val_files, num_parallel_reads=3)
        self.train_raw_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=3)
        self.train_aug_pipeline = train_aug
        self.val_aug_pipeline = val_aug
        self.train_size = train_size
        self.val_size = val_size
        self.img_size = img_size
        self.normalize_image = normalize_image

    def _parse_train_image_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)

        # image functions
        image = tf.image.decode_jpeg(parsed_data['image/encoded'], channels=3)

        if self.train_aug_pipeline != None:
            aug_img = tf.numpy_function(func=self._train_aug_fn, inp=[image], Tout=tf.uint8)
        else:
            aug_img = image

        if self.normalize_image:
            aug_img = tf.cast(aug_img / 255.0, tf.float32)

        if self.img_size != None:
            aug_img = tf.image.resize(aug_img, size=[self.img_size, self.img_size])

        lbl = parsed_data['image/class/id'].values

        one_hot = tf.one_hot(lbl, depth=self.effective_ncls)
        one_hot_multi = tf.reduce_max(one_hot, axis=0)

        return [aug_img, one_hot_multi]

    def _parse_val_image_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)

        # image functions
        image = tf.image.decode_jpeg(parsed_data['image/encoded'], channels=3)

        if self.val_aug_pipeline != None:
            aug_img = tf.numpy_function(func=self._val_aug_fn, inp=[image], Tout=tf.uint8)
        else:
            aug_img = image

        if self.normalize_image:
            aug_img = tf.cast(aug_img / 255.0, tf.float32)

        if self.img_size != None:
            aug_img = tf.image.resize(aug_img, size=[self.img_size, self.img_size])

        lbl = parsed_data['image/class/id'].values

        one_hot = tf.one_hot(lbl, depth=self.effective_ncls)
        one_hot_multi = tf.reduce_max(one_hot, axis=0)

        return [aug_img, one_hot_multi]

    def _train_aug_fn(self, image):
        data = {"image": image}
        aug_data = self.train_aug_pipeline(**data)
        aug_img = aug_data["image"]

        return aug_img

    def train_generator(self, batchsize):
        if self.repeatds:
            tf_trainds = self.train_raw_ds.repeat()
        else:
            tf_trainds = self.train_raw_ds
        tf_trainds = tf_trainds.shuffle(buffer_size=1024)
        tf_trainds = tf_trainds.map(self._parse_train_image_function, num_parallel_calls=AUTOTUNE)

        tf_trainds = tf_trainds.prefetch(AUTOTUNE)
        # dataset_itr = tf_trainds.batch(batchsize, drop_remainder=False)
        return tf_trainds

    def _val_aug_fn(self, image):
        data = {"image": image}
        aug_data = self.val_aug_pipeline(**data)
        aug_img = aug_data["image"]

        return aug_img

    def validation_generator(self, batchsize):
        if self.repeatds:
            tf_valds = self.val_raw_ds.repeat()
        else:
            tf_valds = self.val_raw_ds
        tf_valds = tf_valds.shuffle(buffer_size=1024)
        tf_valds = tf_valds.map(self._parse_val_image_function, num_parallel_calls=AUTOTUNE)
        tf_valds = tf_valds.prefetch(AUTOTUNE)
        # dataset_itr = tf_valds.batch(batchsize, drop_remainder=False)
        return tf_valds