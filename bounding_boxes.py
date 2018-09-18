from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path
import sys
import xml.etree.ElementTree as ET
from scipy.misc import imread,imresize,imsave
from PIL import Image


class BoundingBox(object):
  pass


def GetItem(name, root, index=0):
  count = 0
  for item in root.iter(name):
    if count == index:
      return item.text
    count += 1
  # Failed to find "index" occurrence of item.
  return -1


def GetInt(name, root, index=0):
  return int(GetItem(name, root, index))


def FindNumberBoundingBoxes(root):
  index = 0
  while True:
    if GetInt('xmin', root, index) == -1:
      break
    index += 1
  return index


def ProcessXMLAnnotation(xml_file):
  """Process a single XML file containing a bounding box."""
  # pylint: disable=broad-except
  try:
    tree = ET.parse(xml_file)
  except Exception:
    print('Failed to parse: ' + xml_file, file=sys.stderr)
    return None
  # pylint: enable=broad-except
  root = tree.getroot()

  num_boxes = FindNumberBoundingBoxes(root)
  boxes = []
  objects = tree.findall('object')
  for object_iter in objects:
     bndbox = object_iter.find("bndbox")
     boxes.append([int(it.text) for it in bndbox])
  
  return boxes

if __name__ == '__main__':
  if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('Invalid usage\n'
          'usage: process_bounding_boxes.py <dir> [synsets-file]',
          file=sys.stderr)
    sys.exit(-1)

  xml_files = glob.glob(sys.argv[1] + '/*.xml')
  print('Identified %d XML files in %s' % (len(xml_files), sys.argv[1]),
        file=sys.stderr)

  if len(sys.argv) == 3:
    labels = set([l.strip() for l in open(sys.argv[2]).readlines()])
    print('Identified %d synset IDs in %s' % (len(labels), sys.argv[2]),
          file=sys.stderr)
  else:
    labels = None

  skipped_boxes = 0
  skipped_files = 0
  saved_boxes = 0
  saved_files = 0
  for file_index, one_file in enumerate(xml_files):
    base = os.path.basename(one_file)
    label = os.path.splitext(base)[0]
    print(label)
    # Determine if the annotation is from an ImageNet Challenge label.
    if labels is not None and label not in labels:
      skipped_files += 1
      continue

    bboxes = ProcessXMLAnnotation(one_file)
    assert bboxes is not None, 'No bounding boxes found in ' + one_file

    found_box = False
    for bbox in bboxes:
      image_filename = os.path.splitext(os.path.basename(one_file))[0]
      imgPath = "images/"+image_filename+".jpg"
      im = Image.open(imgPath)
      newim = im.crop(bbox)
      imgPath1 = "images_new/"+image_filename+".jpg"
      imsave(imgPath1,newim)
      saved_boxes += 1
      found_box = True
    if found_box:
      saved_files += 1
    else:
      skipped_files += 1

    if not file_index % 5000:
      print('--> processed %d of %d XML files.' %
            (file_index + 1, len(xml_files)),
            file=sys.stderr)
      print('--> skipped %d boxes and %d XML files.' %
            (skipped_boxes, skipped_files), file=sys.stderr)

  print('Finished processing %d XML files.' % len(xml_files), file=sys.stderr)
  print('Skipped %d XML files not in ImageNet Challenge.' % skipped_files,
        file=sys.stderr)
  print('Skipped %d bounding boxes not in ImageNet Challenge.' % skipped_boxes,
        file=sys.stderr)
  print('Wrote %d bounding boxes from %d annotated images.' %
        (saved_boxes, saved_files),
        file=sys.stderr)
  print('Finished.', file=sys.stderr)
