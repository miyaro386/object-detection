import numpy as np
import os
import sys
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append("..")
from object_detection.utils import ops as utils_ops
# from utils import ops as utils_ops
from glob import glob
from utils import visualization_utils as vis_util
import cv2

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


def build_graph(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def load_image_for_detection(image_path):
    try:
        image_np = np.copy(cv2.imread(image_path))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = image_np.astype(np.uint8)
    except Exception as e:
        print("catch exception", e)

    return image_np


def filter_lsun_dataset(args):
    category_name = args.category_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    TEST_IMAGE_PATHS = list()
    TEST_IMAGE_PATHS += glob(input_dir + "/*.webp")

    print(len(TEST_IMAGE_PATHS), "images")
    detection_graph = build_graph(PATH_TO_CKPT)

    for index, image_path in tqdm(enumerate(TEST_IMAGE_PATHS)):

        _, file_name = os.path.split(image_path)
        file_name = file_name.replace(".webp", ".png")
        bb_file_name = file_name.replace(".png", ".txt")
        output_file_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_file_path):
            print("pass", output_file_path)
            continue

        print("detecting", image_path)
        image_np = load_image_for_detection(image_path)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        save_flag = vis_util.check_boxes_and_labels_on_image_array(
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            category_name=category_name,
            min_score_thresh=0.8,
            size_threshold=0.2,
            edge_threshold=-10.0,
            skip_scores=False,
            instance_masks=output_dict.get('detection_masks'),
        )

        if save_flag:
            print("save image", output_file_path)
            plt.imsave(output_file_path, image_np)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--category_name', type=str, default='car', help='choose in ["couch", "chair", "car"]')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    filter_lsun_dataset(args)

