print("******* Started *******\n")
import  numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2

sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'


PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)

cap = cv2.VideoCapture('test2.mp4') #Change Input File

if (cap.isOpened()== False):
  print("Error opening video  file")

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == True:
        screen = cv2.resize(frame, (800,450))
        image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes1, scores1, classes1, num_detections1) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes1),np.squeeze(classes1).astype(np.int32),np.squeeze(scores1),category_index,use_normalized_coordinates=True,line_thickness=8)

        for i,b in enumerate(boxes1[0]):
                #                 car                    bus                  truck
                if classes1[0][i] == 3 or classes1[0][i] == 6 or classes1[0][i] == 8:
                        if scores1[0][i] >= 0.5:
                                mid_x = (boxes1[0][i][1]+boxes1[0][i][3])/2
                                mid_y = (boxes1[0][i][0]+boxes1[0][i][2])/2
                                apx_distance = round(((1 - (boxes1[0][i][3] - boxes1[0][i][1]))**4),1)
                                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                                if apx_distance <=0.5:
                                        if mid_x > 0.3 and mid_x < 0.7:
                                                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        
        cv2.imshow('window',cv2.resize(image_np,(800,450)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
                break

      else:
              break
        
    cap.release()
    cv2.destroyAllWindows()
    sess.close()
