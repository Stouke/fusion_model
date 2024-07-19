import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load preprocessed data for testing
test_images = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_images.npy')
test_depth_maps = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_depth_maps.npy')
test_reflectance_maps = np.load(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data\test_reflectance_maps.npy')

# Load the trained model
model = load_model(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\trained_fusion_model.h5')

# Run inference on the test data starting from frame 193
start_frame = 193
predictions = model.predict([test_images[start_frame:], test_depth_maps[start_frame:], test_reflectance_maps[start_frame:]])

bbox_predictions = predictions[0]
class_predictions = predictions[1]

# Define a function to visualize the results
def visualize_predictions(image, bbox, score, threshold=0.5):
    if score < threshold:
        return image

    img_height, img_width, _ = image.shape

    # Assuming bbox is in the format (center_x, center_y, width, height) with normalized values
    center_x, center_y, width, height = bbox

    # Print the raw bounding box values for debugging
    print(f'Raw bounding box values: center_x={center_x}, center_y={center_y}, width={width}, height={height}')

    # Convert normalized coordinates to pixel coordinates
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height

    # Ensure width and height are positive
    width = abs(width)
    height = abs(height)

    # Calculate x_min, y_min, x_max, y_max
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)

    print(f'Calculated bounding box coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}')

    # Ensure the coordinates are within the image boundaries
    x_min = max(0, min(x_min, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    x_max = max(0, min(x_max, img_width - 1))
    y_max = max(0, min(y_max, img_height - 1))

    print(f'Adjusted bounding box coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}')

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    label_text = f'Pedestrian: {score:.2f}'
    cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Visualize predictions for a few test images starting from frame 193
num_visualizations = min(20, len(test_images[start_frame:]))  # Visualize at most 10 images

for i in range(num_visualizations):
    img = test_images[start_frame + i]
    bbox = bbox_predictions[i]
    score = class_predictions[i]

    img_copy = img.copy()  # Make a copy to avoid modifying the original image

    if isinstance(score, np.ndarray):
        score = score[0]  # Assuming the score is a one-element array, take the scalar value

    print(f'Image index: {start_frame + i}, Predicted score: {score}')
    visualized_img = visualize_predictions(img_copy, bbox, score)  # Assuming label 1 for pedestrian

    # Convert BGR to RGB for correct display
    visualized_img_rgb = cv2.cvtColor(visualized_img, cv2.COLOR_BGR2RGB)
    
    cv2.imshow(f'Predicted Score: {score:.2f}', visualized_img_rgb)

    # Wait for 'q' key to be pressed to close the window
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()  # Ensure all windows are closed at the end
