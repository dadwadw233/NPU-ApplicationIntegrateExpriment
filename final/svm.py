import numpy as np
import gzip
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import pickle

TRAIN_DATA_PATH = "mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS_PATH = "mnist/train-labels-idx1-ubyte.gz"
TEST_DATA_PATH = "mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_PATH = "mnist/t10k-labels-idx1-ubyte.gz"
MODEL_SAVE = "evaluate/svm_linear.model"


class Kernel:
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def polynomial(degree=3, r=1):
        return lambda x, y: (np.inner(x, y) + r) ** degree

    @staticmethod
    def rbf(gamma=0.01):
        return lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    @staticmethod
    def sigmoid(a=1, r=1):
        return lambda x, y: np.tanh(a * np.inner(x, y) + r)


class SVM:
    def __init__(self, C=1.0, kernel=Kernel.linear(), max_iters=1000):
        self.C = C
        self.kernel = kernel
        self.max_iters = max_iters
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.b = 0

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        self.alpha = np.zeros(n_samples)
        for epoch in range(self.max_iters):
            for i in range(n_samples):
                s = 0
                for j in range(n_samples):
                    s += self.alpha[j] * y[j] * K[i, j]
                if y[i] * s < 1:
                    self.alpha[i] += self.C
                else:
                    self.alpha[i] = max(self.alpha[i] - 1 / self.max_iters, 0)

        self.sv = np.where(self.alpha > 1e-5)[0]
        self.alpha = self.alpha[self.sv]
        self.sv_y = y[self.sv]
        for i in range(len(self.alpha)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.alpha * self.sv_y * K[self.sv[i], self.sv])
        self.b /= len(self.alpha)

    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return np.sign(y_predict + self.b)


class MCSVM:
    def __init__(self, C=1.0, kernel=Kernel.linear(), max_iters=1000):
        self.C = C
        self.kernel = kernel
        self.max_iters = max_iters
        self.svms = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            print(f"fitting class {c}")
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(C=self.C, kernel=self.kernel, max_iters=self.max_iters)
            svm.fit(X, y_binary)
            self.svms.append(svm)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes)))
        for idx, svm in enumerate(self.svms):
            predictions[:, idx] = svm.predict(X)
        return self.classes[np.argmax(predictions, axis=1)]


def load_mnist_data(data_path, labels_path):
    with gzip.open(data_path, "rb") as data_file, gzip.open(
        labels_path, "rb"
    ) as labels_file:
        data_file.read(16)
        buf = data_file.read()
        X = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X = X.reshape(-1, 28 * 28)
        labels_file.read(8)
        buf = labels_file.read()
        y = np.frombuffer(buf, dtype=np.uint8).astype(int)
    X = X / 255.0
    return X, y


def visualize_predictions(svm, x, y, num_samples=10):
    if not os.path.exists("evaluate"):
        os.mkdir("evaluate")

    indices = np.random.choice(x.shape[0], num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

    for idx, ax in zip(indices, axes):
        predicted_label = svm.predict(x[idx].reshape(1, -1))
        true_label = y[idx]

        ax.imshow(x[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("evaluate/svm_visualized_predictions.png")
    plt.show()


def shrink_dataset(X, y, samples_per_class=1000):
    sampled_data = {i: [] for i in range(10)}
    for i in range(len(X)):
        label = y[i]
        if len(sampled_data[label]) < samples_per_class:
            sampled_data[label].append(X[i])
    sampled_X = []
    sampled_y = []
    for label, samples in sampled_data.items():
        sampled_X.extend(samples)
        sampled_y.extend([label] * len(samples))
    X = np.array(sampled_X)
    y = np.array(sampled_y)
    return X, y


if __name__ == "__main__":
    print("Loading Training Data...")
    X_train, y_train = load_mnist_data(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
    # X_train, y_train = shrink_dataset(X_train, y_train, 1000)
    print("Loading Test Data...")
    X_test, y_test = load_mnist_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    # X_test, y_test = shrink_dataset(X_test, y_test, 200)
    svm = None
    if os.path.exists(MODEL_SAVE):
        print("Loading linear kernel...")
        with open(MODEL_SAVE, "rb") as fp:
            svm = pickle.load(fp)
    else:
        print("Training with linear kernel...")
        svm = MCSVM(kernel=Kernel.linear(), C=1)
        svm.fit(X_train, y_train)
        with open(MODEL_SAVE, "wb") as fp:
            pickle.dump(svm, fp)
    pred = svm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{acc * 100:.2f}%")
    visualize_predictions(svm, X_test, y_test)

    print("Training with different C values...")
    acc = []
    c_candidates = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000]
    for c in c_candidates:
        svm = MCSVM(kernel=Kernel.linear(), C=c)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)
        print(f"{a * 100:.2f}%")
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(c_candidates, acc, "-D", color="green", label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Cost Parameter C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy-Cost Parameter")
    plt.savefig("evaluate/svm_ACC_C.png")
    plt.show()

    print("Training with gaussian kernel...")
    acc = []
    g_candidates = [0.01, 0.1, 0.5, 1, 5, 10, 100]
    for g in g_candidates:
        svm = MCSVM(kernel=Kernel.rbf(gamma=g), C=1)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)
        print(f"{a * 100:.2f}%")
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(g_candidates, acc, "-D", color="blue", label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Gamma (with C=1)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy-Gamma")
    plt.savefig("evaluate/svm_ACC_G.png")
    plt.show()

    print("Training with polynomial kernel...")
    acc = []
    d_candidates = [1, 2, 3, 4, 5, 6]
    for d in d_candidates:
        svm = MCSVM(kernel=Kernel.polynomial(degree=d), C=1)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)
        print(f"{a * 100:.2f}%")
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(d_candidates, acc, "-D", color="red", label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Degree (with C=1)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy-Degree")
    plt.savefig("evaluate/svm_ACC_D.png")
    plt.show()
