import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("dataset: mnist")
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


def visualize_network_structure(layer_sizes, save_path="./evaluate/network_structure.png"):

    fig, ax = plt.subplots(figsize=(10, 6))
    n_layers = len(layer_sizes)
    
    width = 0.6  
    gap = 0.8   

    for n, layer_size in enumerate(layer_sizes):
        rect_height = layer_size / max(layer_sizes) * width
        rect_start = 0.5 - rect_height / 2

        rectangle = plt.Rectangle((n * (width + gap), rect_start), width, rect_height,
                                  color='skyblue', ec='black')
        ax.add_artist(rectangle)

        ax.text(n * (width + gap) + width / 2, 0.5 - 0.1, f"Layer {n+1}\n{layer_size} neurons",
                ha='center', va='center', fontsize=10)

    ax.set_xlim(-gap, n * (width + gap) + width)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    print(f"Saving network structure visualization to {save_path}")
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def visualize_predictions(nn, x, y, num_samples=10):
    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')

    indices = np.random.choice(x.shape[0], num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

    for idx, ax in zip(indices, axes):
        prediction = nn.forward(x[idx].reshape(1, -1))
        predicted_label = np.argmax(prediction)
        true_label = np.argmax(y[idx])

        ax.imshow(x[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('evaluate/visualized_predictions.png')
    plt.show()

def visualize_backpropagation(save_path="evaluate/backpropagation_visualization.png"):
    fig, ax = plt.subplots(figsize=(12, 7))


    layers = [(1, 4), (2, 5), (2, 3), (2, 1), (3, 4)]
    sizes = [0.3, 0.3, 0.3, 0.3, 0.3]

    for node, size in zip(layers, sizes):
        circle = plt.Circle(node, size, color='lightblue', ec='black')
        ax.add_artist(circle)


    labels = ["Input", "Hidden 1", "Hidden 2", "Output", "Loss"]
    for i, node in enumerate(layers):
        ax.text(node[0], node[1], labels[i], ha='center', va='center', fontsize=10)


    arrows = [(layers[0], layers[1]), (layers[1], layers[2]), (layers[2], layers[3]), (layers[3], layers[4])]
    for arrow in arrows:
        ax.annotate("", xy=arrow[1], xytext=arrow[0], arrowprops=dict(arrowstyle="->", lw=1))


    back_arrows = [(layers[4], layers[3]), (layers[3], layers[2]), (layers[2], layers[1]), (layers[1], layers[0])]
    gradient_labels = [r"$\frac{\partial L}{\partial o}$", r"$\frac{\partial L}{\partial h2}$",
                       r"$\frac{\partial L}{\partial h1}$", r"$\frac{\partial L}{\partial i}$"]

    for i, arrow in enumerate(back_arrows):
        ax.annotate(gradient_labels[i], xy=arrow[1], xytext=arrow[0],
                    arrowprops=dict(arrowstyle="->", color='red', lw=1), color='red', fontsize=14, 
                    va='center', ha='center')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.axis('off')

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))
        self.loss_function = None
        self.loss_derivative = None
        self.activate_function = None
        self.activate_function_derivative = None
        self.loss_history = []


    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    @staticmethod
    def Elu(x):
        return np.where(x > 0, x, np.exp(x) - 1)
    
    @staticmethod
    def Elu_derivative(x):
        return np.where(x > 0, 1, np.exp(x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))
    
    @staticmethod
    def leaky_relu(x):
        return np.where(x > 0, x, x * 0.01)
    
    @staticmethod
    def leaky_relu_derivative(x):
        return np.where(x > 0, 1, 0.01)

    @staticmethod
    def cross_entropy(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce_loss = -np.sum(targets * np.log(predictions + 1e-9)) / N
        return ce_loss
    
    @staticmethod
    def cross_entropy_derivative(predictions, targets):
        return (predictions - targets) / (predictions * (1 - predictions))
    
    @staticmethod
    def mse(predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mse_derivative(predictions, targets):
        return 2 * (predictions - targets) / targets.size
    
    @staticmethod
    def l1_loss(predictions, targets):
        return np.mean(np.abs(predictions - targets))
    
    @staticmethod
    def l1_loss_derivative(predictions, targets):
        return np.sign(predictions - targets)
    

    @staticmethod
    def l2_loss(predictions, targets):
        return np.mean((predictions - targets) ** 2) / 2
    
    @staticmethod
    def l2_loss_derivative(predictions, targets):
        return (predictions - targets) / targets.size
    
    @staticmethod
    def huber_loss(predictions, targets, delta=1.0):
        abs_error = np.abs(predictions - targets)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)
    
    @staticmethod
    def huber_loss_derivative(predictions, targets, delta=1.0):
        error = predictions - targets
        abs_error = np.abs(error)
        return np.where(abs_error <= delta, error, delta * np.sign(error))
    
    @staticmethod
    def smooth_l1_loss(predictions, targets, delta=1.0):
        abs_error = np.abs(predictions - targets)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)
    
    @staticmethod
    def smooth_l1_loss_derivative(predictions, targets, delta=1.0):
        error = predictions - targets
        abs_error = np.abs(error)
        return np.where(abs_error <= delta, error, delta * np.sign(error))
    
    @staticmethod
    def kl_divergence(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        targets = np.clip(targets, epsilon, 1. - epsilon)
        return np.sum(targets * np.log(targets / predictions), axis=1)
    
    @staticmethod
    def warmup_lr(epoch, lr):
        if epoch < 5:
            return lr * (epoch + 1) / 5
        return lr
    
    @staticmethod
    def lr_decay(epoch, lr):
        # strategy: linear decay
        return lr * 0.1 ** (epoch // 10)
    
    @staticmethod
    def lr_decay2(epoch, lr):
        return lr / (1 + epoch * 0.1)
    
    @staticmethod
    def lr_decay3(epoch, lr):
        return lr * np.exp(-0.1 * epoch)
    
    

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, x, y, learning_rate):
        m = y.shape[0]

        dz3 = self.a3 - y
        dw3 = 1/m * np.dot(self.a2.T, dz3)
        db3 = 1/m * np.sum(dz3, axis=0, keepdims=True)

        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.a2)
        dw2 = 1/m * np.dot(self.a1.T, dz2)
        db2 = 1/m * np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.a1)
        dw1 = 1/m * np.dot(x.T, dz1)
        db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)


        self.W3 -= learning_rate * dw3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
    
    def set_loss_function(self, loss_function):
        if loss_function == "cross_entropy":
            self.loss_function = self.cross_entropy
            self.loss_derivative = self.cross_entropy_derivative
        elif loss_function == "mse":
            self.loss_function = self.mse
            self.loss_derivative = self.mse_derivative
        elif loss_function == "l1_loss":
            self.loss_function = self.l1_loss
            self.loss_derivative = self.l1_loss_derivative
        elif loss_function == "l2_loss":
            self.loss_function = self.l2_loss
            self.loss_derivative = self.l2_loss_derivative
        elif loss_function == "huber_loss":
            self.loss_function = self.huber_loss
            self.loss_derivative = self.huber_loss_derivative
        elif loss_function == "smooth_l1_loss":
            self.loss_function = self.smooth_l1_loss
            self.loss_derivative = self.smooth_l1_loss_derivative
        elif loss_function == "kl_divergence":
            self.loss_function = self.kl_divergence
            self.loss_derivative = None
        else:
            raise ValueError(f"Unknown loss function {loss_function}")

    def set_activate_function(self, activate_function):
        if activate_function == "relu":
            self.activate_function = self.relu
            self.activate_function_derivative = self.relu_derivative
        elif activate_function == "Elu":
            self.activate_function = self.Elu
            self.activate_function_derivative = self.Elu_derivative
        elif activate_function == "tanh":
            self.activate_function = self.tanh
            self.activate_function_derivative = self.tanh_derivative
        elif activate_function == "sigmoid":
            self.activate_function = self.sigmoid
            self.activate_function_derivative = self.sigmoid_derivative
        elif activate_function == "leaky_relu":
            self.activate_function = self.leaky_relu
            self.activate_function_derivative = self.leaky_relu_derivative
        else:
            raise ValueError(f"Unknown activate function {activate_function}")
        

    def summary(self):
        print("Layer 1")
        print(f"Weight shape: {self.W1.shape}")
        print(f"Bias shape: {self.b1.shape}")
        print("Layer 2")
        print(f"Weight shape: {self.W2.shape}")
        print(f"Bias shape: {self.b2.shape}")
        print("Layer 3")
        print(f"Weight shape: {self.W3.shape}")
        print(f"Bias shape: {self.b3.shape}")

    def predict(self, x):
        return self.forward(x)
    
    def save_parameters(self, save_path="./evaluate/parameters.npz"):
        np.savez(save_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load_parameters(self, load_path="./evaluate/parameters.npz"):
        parameters = np.load(load_path)
        try: 
            self.W1 = parameters['W1']
            self.b1 = parameters['b1']
            self.W2 = parameters['W2']
            self.b2 = parameters['b2']
            self.W3 = parameters['W3']
            self.b3 = parameters['b3']
        except KeyError:
            raise ValueError(f"Cannot find parameters in {load_path}")
        except Exception as e:
            raise e
        
    def visualize_loss(self, save_path="./evaluate/loss.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.loss_history, label='loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    
    def train(self, x, y, learning_rate, epochs):
        '''
        @param:

            x: input data
            y: labels
            learning_rate: learning rate
            epochs: number of epochs

        '''
        if self.loss_function is None:
            raise ValueError("Loss function not set")
        if self.activate_function is None:
            raise ValueError("Activate function not set")
        for i in range(epochs):
            predictions = self.forward(x)
            loss = self.loss_function(predictions, y)
            self.loss_history.append(loss)
            self.backward(x, y, learning_rate)
            
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {loss:.5f}")
                print(f"Learning rate: {learning_rate:.5f}")
                #learning_rate = self.lr_decay2(i, learning_rate)
                print(f"accuracy: {np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)):.4f}")


if __name__ == "__main__":
    nn = NeuralNetwork(784, 256, 128, 10)
    nn.set_loss_function("cross_entropy")
    nn.set_activate_function("leaky_relu")
    nn.train(x_train, y_train, learning_rate=0.1, epochs=100)
    nn.save_parameters()
    nn.summary()
    nn.load_parameters()
    test_predictions = nn.predict(x_test)
    test_predictions = np.argmax(test_predictions, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(test_predictions == test_labels)
    print(f"Accuracy: {accuracy:.4f}")
    nn.visualize_loss()
    visualize_predictions(nn, x_test, y_test)
    visualize_network_structure([784, 256, 128, 10])
    visualize_backpropagation()