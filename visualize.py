import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

# Set the backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

# Load preprocessed data for visualization
images = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_images.npy')
depth_maps = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_depth_maps.npy')
reflectance_maps = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_reflectance_maps.npy')

with open(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_bboxes.pkl', 'rb') as f:
    bboxes = pickle.load(f)

with open(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\best_model.keras')

# Define a function to visualize bounding boxes
def visualize_bboxes(image, gt_bbox, gt_label, pred_bbox=None, pred_label=None):
    # Draw ground truth bounding box
    x_min, y_min, x_max, y_max = gt_bbox
    if x_min >= 0 and y_min >= 0 and x_max >= 0 and y_max >= 0:
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        label_text = 'Pedestrian' if gt_label == 1 else 'Other'
        cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw predicted bounding box
    if pred_bbox is not None:
        px_min, py_min, px_max, py_max = pred_bbox
        cv2.rectangle(image, (int(px_min), int(py_min)), (int(px_max), int(py_max)), (0, 255, 0), 2)
        pred_label_text = 'Pedestrian' if pred_label >= 0.5 else 'Other'
        cv2.putText(image, pred_label_text, (int(px_min), int(py_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Predict and visualize results for a few test images
num_test_images = len(images)
num_visualizations = min(10, num_test_images)  # Visualize at most 10 images

for i in range(num_visualizations):
    img = np.expand_dims(images[i], axis=0)  # Add batch dimension
    depth = np.expand_dims(depth_maps[i], axis=0)
    reflectance = np.expand_dims(reflectance_maps[i], axis=0)
    gt_bbox = bboxes[i][0] if bboxes[i] else [0, 0, 0, 0]  # Assume one bounding box per image for simplicity
    gt_label = labels[i][0] if labels[i] else 0  # Assume one label per image for simplicity

    # Predict using the model
    pred_bbox, pred_label = model.predict([img, depth, reflectance], verbose=0)
    pred_bbox = pred_bbox[0]
    pred_label = pred_label[0]

    print(f'Image index: {i}, Ground Truth bbox: {gt_bbox}, Ground Truth label: {gt_label}, Predicted bbox: {pred_bbox}, Predicted label: {pred_label}')  # Debugging statement

    visualized_img = visualize_bboxes(images[i], gt_bbox, gt_label, pred_bbox, pred_label)  # Visualize ground truth and prediction

    plt.figure(figsize=(8, 8))
    plt.imshow(visualized_img)
    plt.title(f'Ground Truth and Predicted Bounding Box')
    plt.axis('off')
    plt.savefig(f'predictions/prediction_{i}.png')  # Save the plot as an image file instead of showing it
    plt.close()

print("Prediction visualization completed.")
