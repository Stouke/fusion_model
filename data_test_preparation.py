import numpy as np
import tensorflow as tf
from pykitti import raw
import os
import cv2
import xml.etree.ElementTree as ET
import pickle

def parse_tracklet_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for tracklet in root.findall('.//tracklets/item'):
        object_type_element = tracklet.find('objectType')
        object_type = object_type_element.text if object_type_element is not None else None
        print(f"Found object type: {object_type}")  # Debugging line

        if object_type != 'Pedestrian':
            continue
        
        first_frame = int(tracklet.find('first_frame').text) if tracklet.find('first_frame') is not None else None
        poses = tracklet.find('poses')
        
        pose_items = []
        if poses is not None:
            for pose in poses.findall('item'):
                tx = float(pose.find('tx').text) if pose.find('tx') is not None else None
                ty = float(pose.find('ty').text) if pose.find('ty') is not None else None
                tz = float(pose.find('tz').text) if pose.find('tz') is not None else None
                rx = float(pose.find('rx').text) if pose.find('rx') is not None else None
                ry = float(pose.find('ry').text) if pose.find('ry') is not None else None
                rz = float(pose.find('rz').text) if pose.find('rz') is not None else None
                state = int(pose.find('state').text) if pose.find('state') is not None else None
                occlusion = int(pose.find('occlusion').text) if pose.find('occlusion') is not None else None
                truncation = int(pose.find('truncation').text) if pose.find('truncation') is not None else None

                pose_item = {
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                    'rx': rx,
                    'ry': ry,
                    'rz': rz,
                    'state': state,
                    'occlusion': occlusion,
                    'truncation': truncation
                }
                pose_items.append(pose_item)
        
        annotation = {
            'type': object_type,
            'first_frame': first_frame,
            'poses': pose_items
        }
        annotations.append(annotation)

    return annotations

def point_cloud_to_maps(point_cloud, image_shape):
    depth_map = np.zeros(image_shape)
    reflectance_map = np.zeros(image_shape)
    
    for point in point_cloud:
        x, y, z, reflectance = point
        if z > 0:
            u = int((x * 0.54) / z + image_shape[1] / 2)
            v = int((y * 0.54) / z + image_shape[0] / 2)
            if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                depth_map[v, u] = z
                reflectance_map[v, u] = reflectance
    return depth_map, reflectance_map

def preprocess_frame(frame_idx, dataset, annotations, image_shape=(416, 416)):
    print(f"Preprocessing frame {frame_idx}...")

    image = dataset.get_cam2(frame_idx)
    image = np.array(image)
    image = cv2.resize(image, image_shape)

    point_cloud = dataset.get_velo(frame_idx)
    depth_map, reflectance_map = point_cloud_to_maps(point_cloud, image_shape)

    frame_annotations = [ann for ann in annotations if ann['first_frame'] <= frame_idx < ann['first_frame'] + len(ann['poses'])]

    bounding_boxes = []
    labels = []
    for ann in frame_annotations:
        pose_idx = frame_idx - ann['first_frame']
        if pose_idx < len(ann['poses']):
            pose = ann['poses'][pose_idx]
            bbox = (pose['tx'], pose['ty'], pose['tz'])  # You will need to convert this to pixel coordinates
            bounding_boxes.append(bbox)
            labels.append(1)  # Assuming 1 is the label for Pedestrian

    print(f"Image shape: {image.shape}, Depth map shape: {depth_map.shape}, Reflectance map shape: {reflectance_map.shape}")
    print(f"Bounding boxes: {bounding_boxes}, Labels: {labels}")

    return image, depth_map, reflectance_map, bounding_boxes, labels

def prepare_dataset(basedir, date, drive, xml_file):
    # Load the dataset
    dataset = raw(basedir, date, drive)
    annotations = parse_tracklet_xml(xml_file)

    num_frames = len(dataset.timestamps)
    images, depth_maps, reflectance_maps = [], [], []
    bboxes, labels = [], []

    for frame_idx in range(num_frames):
        image, depth_map, reflectance_map, bounding_boxes, frame_labels = preprocess_frame(frame_idx, dataset, annotations)
        images.append(image)
        depth_maps.append(depth_map)
        reflectance_maps.append(reflectance_map)
        bboxes.append(bounding_boxes)
        labels.append(frame_labels)

    images = np.array(images)
    depth_maps = np.array(depth_maps)
    reflectance_maps = np.array(reflectance_maps)
    # bboxes and labels will remain lists of lists

    print(f"Total images: {len(images)}, Total depth maps: {len(depth_maps)}, Total reflectance maps: {len(reflectance_maps)}")
    print(f"Total bounding boxes: {len(bboxes)}, Total labels: {len(labels)}")

    return images, depth_maps, reflectance_maps, bboxes, labels

# Main execution
basedir = 'C:/Users/n2309064h/Desktop/Multimodal_code/kitti'
date = '2011_09_26'
drive = '0014'
xml_file = r'C:\Users\n2309064h\Desktop\Multimodal_code\kitti\2011_09_26\2011_09_26_drive_0014_sync\tracklet_labels.xml'

images, depth_maps, reflectance_maps, bboxes, labels = prepare_dataset(basedir, date, drive, xml_file)

import os
import numpy as np
import pickle

# Define the base path using os.path.join to handle path separators
base_path = r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data'

# Ensure the directory exists
os.makedirs(base_path, exist_ok=True)

# Save preprocessed data
np.save(os.path.join(base_path, 'test_images.npy'), images)
np.save(os.path.join(base_path, 'test_depth_maps.npy'), depth_maps)
np.save(os.path.join(base_path, 'test_reflectance_maps.npy'), reflectance_maps)

# Save bboxes and labels using pickle
with open(os.path.join(base_path, 'test_bboxes.pkl'), 'wb') as f:
    pickle.dump(bboxes, f)

with open(os.path.join(base_path, 'test_labels.pkl'), 'wb') as f:
    pickle.dump(labels, f)

print("Data preparation completed.")
