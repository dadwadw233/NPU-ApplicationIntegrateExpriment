import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# 加载并预处理数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("dataset: mnist")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255

# to one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("model: CNN")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
print("loss function: categorical_crossentropy")
print("optimizer: adam")
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

def visualize_predictions(model, x, y, num_samples=10):
    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')

    indices = np.random.choice(x.shape[0], num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

    for idx, ax in zip(indices, axes):
        prediction = model.predict(x[idx].reshape(1, 28, 28, 1))
        predicted_label = np.argmax(prediction)
        true_label = np.argmax(y[idx])

        ax.imshow(x[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('./evaluate/cnn_visualized_predictions.png')
    plt.show()

visualize_predictions(model, x_test, y_test)


def visualize_network_structure(model, save_path="./evaluate/cnn_network_structure.png"):
    layer_info = []
    for layer in model.layers:
        if 'conv' in layer.name:
            layer_info.append((layer.output_shape[1], 'Conv', layer.kernel_size, layer.input_shape[1:], layer.output_shape[1:]))
        elif 'dense' in layer.name:
            layer_info.append((layer.output_shape[1], 'Dense', None, None, None))

    fig, ax = plt.subplots(figsize=(15, 8))
    
    width = 0.6
    gap = 1.2

    for n, (layer_size, layer_type, kernel_size, input_dim, output_dim) in enumerate(layer_info):
        rect_height = layer_size / max([info[0] for info in layer_info]) * width
        rect_start = 0.5 - rect_height / 2

        color = 'skyblue' if layer_type == 'Conv' else 'lightgreen'
        rectangle = plt.Rectangle((n * (width + gap), rect_start), width, rect_height,
                                  color=color, ec='black')
        ax.add_artist(rectangle)
        
        if kernel_size:
            label = f"{layer_type}\n{layer_size} units\nKernel: {kernel_size[0]}x{kernel_size[1]}"
            dim_label = f"Input: {input_dim[0]}x{input_dim[1]}x{input_dim[2]}\nOutput: {output_dim[0]}x{output_dim[1]}x{output_dim[2]}"
            ax.text(n * (width + gap) + width / 2, 0.5 + 0.35, dim_label,
                    ha='center', va='center', fontsize=8, wrap=True)
        else:
            label = f"{layer_type}\n{layer_size} units"
            
        ax.text(n * (width + gap) + width / 2, 0.5 - 0.2, label,
                ha='center', va='center', fontsize=8, wrap=True)

    ax.set_xlim(-gap, n * (width + gap) + width)
    ax.set_ylim(0, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

visualize_network_structure(model)
