import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import pickle
import os

# Load preprocessed data
base_path = r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\preprocessed_data'
images = np.load(os.path.join(base_path, 'images.npy'))
depth_maps = np.load(os.path.join(base_path, 'depth_maps.npy'))
reflectance_maps = np.load(os.path.join(base_path, 'reflectance_maps.npy'))

with open(os.path.join(base_path, 'bboxes.pkl'), 'rb') as f:
    bboxes = pickle.load(f)

with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
    labels = pickle.load(f)

print(f'Images shape: {images.shape}')
print(f'Depth maps shape: {depth_maps.shape}')
print(f'Reflectance maps shape: {reflectance_maps.shape}')
print(f'Bounding boxes shape: {len(bboxes)}')
print(f'Labels shape: {len(labels)}')

# Normalize depth maps and reflectance maps
depth_maps = depth_maps / np.max(depth_maps)
reflectance_maps = reflectance_maps / np.max(reflectance_maps)

# Ensure each frame has a bounding box, even if empty
all_bboxes = []
all_labels = []
for i in range(len(images)):
    if len(bboxes[i]) == 0:
        all_bboxes.append([0, 0, 0, 0])
        all_labels.append(0)
    else:
        for bbox in bboxes[i]:
            tx, ty, tz = bbox
            x_min, y_min = tx - 0.5, ty - 0.5  # Placeholder conversion
            x_max, y_max = tx + 0.5, ty + 0.5  # Placeholder conversion
            all_bboxes.append([x_min, y_min, x_max, y_max])
            all_labels.append(labels[i][0])

y_train_bboxes = np.array(all_bboxes)
y_train_labels = np.array(all_labels)

print(f'Bounding boxes shape after processing: {y_train_bboxes.shape}')
print(f'Labels shape after processing: {y_train_labels.shape}')

# Debugging: Print the first few bounding boxes and labels
print("Sample bounding boxes:", y_train_bboxes[:5])
print("Sample labels:", y_train_labels[:5])

# Create the model
def create_lidar_feature_extractor(input_shape):
    input_depth = Input(shape=input_shape)
    input_reflectance = Input(shape=input_shape)
    
    def cnn_branch(input):
        x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        return x
    
    depth_features = cnn_branch(input_depth)
    reflectance_features = cnn_branch(input_reflectance)
    
    combined_features = concatenate([depth_features, reflectance_features])
    
    model = Model(inputs=[input_depth, input_reflectance], outputs=combined_features)
    return model

def create_fusion_model(input_shape):
    input_camera = Input(shape=input_shape, name='camera_input')
    cnn_camera = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_camera)
    cnn_camera = MaxPooling2D((2, 2))(cnn_camera)
    cnn_camera = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(cnn_camera)
    cnn_camera = MaxPooling2D((2, 2))(cnn_camera)
    cnn_camera = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(cnn_camera)
    cnn_camera = MaxPooling2D((2, 2))(cnn_camera)
    cnn_camera = Flatten()(cnn_camera)
    cnn_camera = BatchNormalization()(cnn_camera)
    cnn_camera = Dropout(0.5)(cnn_camera)

    lidar_model = create_lidar_feature_extractor((416, 416, 1))

    combined_features = concatenate([cnn_camera, lidar_model.output])
    combined_features = BatchNormalization()(combined_features)
    combined_features = Dropout(0.5)(combined_features)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)

    bbox_output = Dense(4, activation='linear', name='bbox_output')(x)
    class_output = Dense(1, activation='sigmoid', name='class_output')(x)

    model = Model(inputs=[input_camera, lidar_model.input[0], lidar_model.input[1]], outputs=[bbox_output, class_output])
    return model

# Instantiate and compile the model
model = create_fusion_model((416, 416, 3))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'bbox_output': 'mean_squared_error', 'class_output': 'binary_crossentropy'},
              metrics={'bbox_output': 'mse', 'class_output': 'accuracy'})

# Debugging: Print model summary
print("Model Summary:")
model.summary()

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\best_model.keras', monitor='val_loss', save_best_only=True)

# Debugging: Print training data shapes and first few samples
print(f'Images shape: {images.shape}')
print(f'Depth maps shape: {depth_maps.shape}')
print(f'Reflectance maps shape: {reflectance_maps.shape}')
print(f'Bounding boxes shape: {y_train_bboxes.shape}')
print(f'Labels shape: {y_train_labels.shape}')
print('First few images:', images[:1])
print('First few depth maps:', depth_maps[:1])
print('First few reflectance maps:', reflectance_maps[:1])
print('First few bounding boxes:', y_train_bboxes[:1])
print('First few labels:', y_train_labels[:1])

# Initial loss check
initial_loss = model.evaluate(
    [images, depth_maps, reflectance_maps],
    {'bbox_output': y_train_bboxes, 'class_output': y_train_labels}
)
print(f'Initial loss: {initial_loss}')

# Train the model
history = model.fit(
    [images, depth_maps, reflectance_maps],
    {'bbox_output': y_train_bboxes, 'class_output': y_train_labels},
    epochs=30,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save(r'C:\Users\n2309064h\Desktop\Multimodal_code\fusion model\trained_fusion_model.keras')

# Evaluate the model
evaluation_results = model.evaluate(
    [images, depth_maps, reflectance_maps],
    {'bbox_output': y_train_bboxes, 'class_output': y_train_labels}
)

# Print evaluation results
print(f'Evaluation results: {evaluation_results}')

# Debugging: Print model predictions on a few samples
sample_predictions = model.predict([images[:5], depth_maps[:5], reflectance_maps[:5]])
print("Sample Predictions (Bounding boxes):", sample_predictions[0])
print("Sample Predictions (Class):", sample_predictions[1])
