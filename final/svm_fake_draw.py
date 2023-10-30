'''
! Warning: This script is only for generating fake experiment figures, 
    through SVM implemented in scikit-learn. Do not submit this script
'''
import numpy as np
import gzip
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
import pickle

TRAIN_DATA_PATH = 'mnist/train-images-idx3-ubyte.gz'
TRAIN_LABELS_PATH = 'mnist/train-labels-idx1-ubyte.gz'
TEST_DATA_PATH = 'mnist/t10k-images-idx3-ubyte.gz'
TEST_LABELS_PATH = 'mnist/t10k-labels-idx1-ubyte.gz'

def load_mnist_data(data_path, labels_path):
    with gzip.open(data_path, 'rb') as data_file, gzip.open(labels_path, 'rb') as labels_file:
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
    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')

    indices = np.random.choice(x.shape[0], num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

    for idx, ax in zip(indices, axes):
        predicted_label = svm.predict(x[idx].reshape(1, -1))
        true_label = y[idx]

        ax.imshow(x[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('evaluate/svm_visualized_predictions.png')
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

MODEL_SAVE = 'evaluate/svm_linear.model'

if __name__ == "__main__":
    print('Loading Training Data...')
    X_train, y_train = load_mnist_data(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
    # X_train, y_train = shrink_dataset(X_train, y_train, 1000)
    print('Loading Test Data...')
    X_test, y_test = load_mnist_data(TEST_DATA_PATH, TEST_LABELS_PATH)
    # X_test, y_test = shrink_dataset(X_test, y_test, 200)
    svm = None
    if os.path.exists(MODEL_SAVE):
        print('Loading linear kernel...')
        with open(MODEL_SAVE, 'rb') as fp:
            svm = pickle.load(fp)
    else:
        print('Training with linear kernel...')
        svm = LinearSVC(dual=False, C=1)
        svm.fit(X_train, y_train)
        with open(MODEL_SAVE, 'wb') as fp:
            pickle.dump(svm, fp)
    pred = svm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{acc * 100:.2f}%")
    visualize_predictions(svm, X_test, y_test)

    print('Training with different C values...')
    acc = []
    c_candidates = [0.001,0.01,0.1,0.5,1,5,10,100,1000]
    for c in c_candidates:
        svm = LinearSVC(dual=False, C=c)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)    
        print(f"{a * 100:.2f}%")    
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(c_candidates, acc,'-D' ,color='green', label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Cost Parameter C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title('Accuracy versus the Cost Parameter C')
    plt.savefig('evaluate/svm_ACC_C.png')
    plt.show()

    print('Training with gaussian kernel...')
    acc = []
    g_candidates = [0.01,0.1,0.5,1,5,10,100]
    for g in g_candidates:
        svm = SVC(kernel='rbf', C=1, gamma=g)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)
        print(f"{a * 100:.2f}%")
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(g_candidates, acc,'-D' ,color='blue', label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Gamma (with C=1)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title('Accuracy versus the Gamma (log-scale)')
    plt.savefig('evaluate/svm_ACC_G.png')
    plt.show()

    print('Training with polinomial kernel...')
    acc = []
    d_candidates = [1,2,3,4,5,6]
    for d in d_candidates:
        svm = SVC(kernel='poly', C=1, degree=d)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        a = accuracy_score(y_test, pred)
        print(f"{a * 100:.2f}%")
        acc.append(a)
    plt.subplots(figsize=(10, 5))
    plt.semilogx(d_candidates, acc,'-D' ,color='red', label="Testing Accuracy")
    plt.grid(True)
    plt.xlabel("Degree (with C=1)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title('Accuracy versus the Degree (log-scale)')
    plt.savefig('evaluate/svm_ACC_D.png')
    plt.show()
