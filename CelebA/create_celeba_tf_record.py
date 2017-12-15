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

r"""Convert CelebA dataset to TFRecord for object_detection.

Example usage:
python create_celeba_tf_record.py --data_dir=/local/mnt2/workspace2/chris/Databases/CelebA \
    --set=train --annotations_dir=Anno --output_path=celeba_train.record

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
flags.DEFINE_string('data_dir', '', 'Root directory to umdfaces dataset.')
flags.DEFINE_string('set', 'train','Convert training set or test set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def dict_to_tf_example(landmark_anno, bbox_anno,
                       dataset_directory):
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
  img_path = os.path.join(dataset_directory, 'img_celeba')
  full_path = os.path.join(img_path, landmark_anno.strip().split(' ')[0])
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

  bbox_number = [int(s) for s in bbox_anno.split() if s.isdigit()]
  landmark_numbers = [int(s) for s in landmark_anno.split() if s.isdigit()]

  xmin.append(float(bbox_number[0]) / width)
  ymin.append(float(bbox_number[1]) / height)
  xmax.append(float(bbox_number[0] + bbox_number[2]) / width)
  ymax.append(float(bbox_number[1] + bbox_number[3]) / height)

  landmark = ""
  visibility = ""
  for i in range(5):
      landmark += str(float(landmark_numbers[i*2+1]) / height)
      landmark += " "
      landmark += str(float(landmark_numbers[i*2]) / width)
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
      'image/filename': dataset_util.bytes_feature(landmark_anno.strip().split(' ')[0]),
      'image/source_id': dataset_util.bytes_feature(landmark_anno.strip().split(' ')[0]),
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

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  examples_path = os.path.join(data_dir, 'list_eval_partition.txt')
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)

  with tf.gfile.GFile(examples_path) as fid:
      lines = fid.readlines()
  images_names = [line.strip().split(' ')[0] for line in lines]
  images_set = [line.strip().split(' ')[1] for line in lines]
  landmark_anno_file = open(os.path.join(annotations_dir, 'list_landmarks_celeba.txt'), 'r')
  bbox_anno_file = open(os.path.join(annotations_dir, 'list_bbox_celeba.txt'), 'r')

  # Discard first two lines
  landmark_anno_file.readline()
  landmark_anno_file.readline()
  bbox_anno_file.readline()
  bbox_anno_file.readline()

  for idx, example in enumerate(images_names):
      landmark_anno = landmark_anno_file.readline()
      bbox_anno = bbox_anno_file.readline()
      if (images_set[idx] == '0' and FLAGS.set == 'train') or (images_set[idx] == '1' and FLAGS.set == 'eval') \
        or (images_set[idx] == '2' and FLAGS.set == 'test'):
          if idx % 100 == 0:
              print('On image %d of %d', idx, len(images_names))
          tf_example = dict_to_tf_example(landmark_anno, bbox_anno, FLAGS.data_dir)
          writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
