# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert FDDB dataset to TFRecord for object_detection.

Example usage:
python create_fddb_tf_record.py --data_dir=/local/mnt2/workspace2/chris/Databases/FDDB \
    --FDDB_fold=FDDB-fold-01 --output_path=FDDB-fold-01.record

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
import PIL.Image
import tensorflow as tf


import sys
sys.path.append('/local/mnt/workspace/chris/projects/models/')

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to FDDB dataset.')
flags.DEFINE_string('FDDB_fold', '',
                    'Specify which fold to generate')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def dict_to_tf_example(image_name, dataset_directory):
  """Convert one line annotation and image to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    landmark_anno: the line containing the landmark annotation
    bbox_anno: the line containing the bbox annotation
    dataset_directory: Path to root directory holding PASCAL dataset

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(dataset_directory, image_name+'.jpg')
  assert os.path.isfile(full_path)
  with tf.gfile.GFile(full_path) as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  landmarks = []
  visibilities =[]

  classes = []
  classes_text = []

  # Dummy groundtruth to match the format needed for input_reader_builder
  xmin.append(0.0)
  ymin.append(0.0)
  xmax.append(0.0)
  ymax.append(0.0)

  landmark = ""
  visibility = ""
  for i in range(5):
      landmark += str(0.0)
      landmark += " "
      landmark += str(0.0)
      landmark += " "
      visibility += '1'
      visibility += " "

  landmarks.append(landmark)
  visibilities.append(visibility)
  classes_text.append('face')
  classes.append(1)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_name),
      'image/source_id': dataset_util.bytes_feature(image_name),
      'image/key/sha256': dataset_util.bytes_feature(key),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),

      'image/object/landmarks': dataset_util.bytes_list_feature(landmarks),
      'image/object/visbilities': dataset_util.bytes_list_feature(visibilities),

  }))
  return example


def main(_):
  data_dir = FLAGS.data_dir
  FDDB_fold = FLAGS.FDDB_fold

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  if (FDDB_fold):
      examples_path = os.path.join(data_dir, 'FDDB-folds', FDDB_fold+'.txt')
      assert os.path.isfile(examples_path)

      with tf.gfile.GFile(examples_path) as fid:
          images_names = fid.readlines()
  else:
      images_names = []
      for i in range(1,11):
          i = format(i, '02')
          examples_path = os.path.join(data_dir, 'FDDB-folds', 'FDDB-fold-'+i+'.txt')
          assert os.path.isfile(examples_path)

          with tf.gfile.GFile(examples_path) as fid:
              images_names.extend(fid.readlines())

  images_names = [images_name[:-1] for images_name in images_names]

  for idx, example in enumerate(images_names):
      if idx % 100 == 0:
          print('On image {} of {}'.format( idx, len(images_names)))

      tf_example = dict_to_tf_example(example, FLAGS.data_dir)
      writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
