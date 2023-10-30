import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import os

TRAIN_DATA_FILE = "train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILE = "train-labels-idx1-ubyte.gz"
TEST_DATA_FILE = "t10k-images-idx3-ubyte.gz"
TEST_LABELS_FILE = "t10k-labels-idx1-ubyte.gz"


def load_mnist(data_dir):
    def _load_mnist_data(data_path, labels_path):
        with gzip.open(data_path, "rb") as data_file:
            data_file.read(16)
            buf = data_file.read()
            X = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            X = X.reshape(-1, 28 * 28)
        with gzip.open(labels_path, "rb") as labels_file:
            labels_file.read(8)
            buf = labels_file.read()
            y = np.frombuffer(buf, dtype=np.uint8).astype(int)
        X = X / 255.0
        return X, y

    x_tr_path = os.path.join(data_dir, TRAIN_DATA_FILE)
    y_tr_path = os.path.join(data_dir, TRAIN_LABELS_FILE)
    x_ts_path = os.path.join(data_dir, TEST_DATA_FILE)
    y_ts_path = os.path.join(data_dir, TEST_LABELS_FILE)
    x_tr, y_tr = _load_mnist_data(x_tr_path, y_tr_path)
    x_ts, y_ts = _load_mnist_data(x_ts_path, y_ts_path)
    return x_tr, y_tr, x_ts, y_ts


def softmax(z):
    exp_list = np.exp(z)
    result = 1 / sum(exp_list) * exp_list
    return result.reshape((len(z), 1))


def neg_log_loss(pred, label):
    return -np.log(pred[int(label)])


def plot_log(loss_list, acc_list, output_path):
    epoch_list = list(range(1, len(loss_list) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(epoch_list)
    plt.title("Loss ~ Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, acc_list)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xticks(epoch_list)
    plt.title("Accuracy ~ Epoch")
    plt.legend()

    plt.tight_layout()
    plt.suptitle(output_path)
    plt.savefig(output_path)
    plt.show()


def plot_predictions(clf, x, y, output_path):
    if not os.path.exists("evaluate"):
        os.mkdir("evaluate")

    num_samples = 10
    indices = np.random.choice(x.shape[0], num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))

    for idx, ax in zip(indices, axes):
        predicted_label = clf.predict(x[idx].reshape(1, -1))
        true_label = y[idx]

        ax.imshow(x[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle(output_path)
    plt.savefig(output_path)
    plt.show()


def plot_confusion(pred, label, output_path):
    plt.subplots(figsize=(10, 6))
    sb.heatmap(confusion_matrix(pred, label), annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.suptitle(output_path)
    plt.savefig(output_path)
    plt.show()


def plot_weights(clf, output_path):
    plt.subplots(2, 5, figsize=(20, 8))
    for i in range(10):
        subplt = plt.subplot(2, 5, i + 1)
        subplt.imshow(
            clf.w[i, :].reshape(28, 28), interpolation="nearest", cmap=plt.cm.RdBu
        )
        subplt.set_xticks(())
        subplt.set_yticks(())
        subplt.set_xlabel("Class %i" % i)
    plt.suptitle(output_path)
    plt.savefig(output_path)
    plt.show()


class LogClf:
    def __init__(
        self,
        n_epoches=15,
        batch_size=10,
        lr=0.005,
        decay=0.75,
        mu=0,
        seed=42,
        verbose=False,
    ) -> None:
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.mu = mu
        self.seed = seed
        self.verbose = verbose

    def mini_batch_gradient(self, x_batch, y_batch):
        batch_size = x_batch.shape[0]
        w_grad_list = []
        b_grad_list = []
        batch_loss = 0
        for i in range(batch_size):
            x, y = x_batch[i], y_batch[i]
            x = x.reshape((784, 1))
            E = np.zeros((10, 1))
            E[y][0] = 1
            pred = softmax(np.matmul(self.w, x) + self.b)

            loss = neg_log_loss(pred, y)
            batch_loss += loss

            w_grad = E - pred
            w_grad = -np.matmul(w_grad, x.reshape((1, 784)))
            w_grad_list.append(w_grad)

            b_grad = -(E - pred)
            b_grad_list.append(b_grad)

        dw = sum(w_grad_list) / batch_size
        db = sum(b_grad_list) / batch_size
        return dw, db, batch_loss

    def fit(self, X, y, X_ts=None, y_ts=None):
        np.random.seed(self.seed)
        n_feat = X.shape[1]
        n_class = len(np.unique(y))
        self.w = np.random.randn(n_class, n_feat) / np.sqrt(n_class * n_feat)
        self.b = np.random.randn(n_class, 1) / np.sqrt(n_class)
        tr_loss_list, tr_acc_list = [], []
        ts_loss_list, ts_acc_list = [], []
        learning_rate = self.lr

        if self.mu is not None:
            w_velocity = np.zeros(self.w.shape)
            b_velocity = np.zeros(self.b.shape)

        for epoch in range(self.n_epoches):
            rand_indices = np.random.choice(X.shape[0], X.shape[0], replace=False)
            num_batch = int(X.shape[0] / self.batch_size)

            if self.decay is not None:
                try:
                    if tr_acc_list[-1] - tr_acc_list[-2] < 0.001:
                        learning_rate *= self.decay
                except:
                    pass

                message = "learning rate: %.8f" % learning_rate
                if self.verbose:
                    print(message)
                logging.info(message)

            for batch in range(num_batch):
                index = rand_indices[
                    self.batch_size * batch : self.batch_size * (batch + 1)
                ]
                x_batch = X[index]
                y_batch = y[index]

                dw, db, batch_loss = self.mini_batch_gradient(x_batch, y_batch)
                if self.mu is not None:
                    w_velocity = self.mu * w_velocity + learning_rate * dw
                    b_velocity = self.mu * b_velocity + learning_rate * db
                    self.w -= w_velocity
                    self.b -= b_velocity
                else:
                    self.w -= learning_rate * dw
                    self.b -= learning_rate * db
                if batch % 100 == 0:
                    message = (
                        f"Epoch {epoch+1}, Batch {batch}, Loss {batch_loss[0]:.2f}"
                    )
                    if self.verbose:
                        print(message)
                    logging.info(message)

            tr_loss, tr_acc = self.evaluate(X, y)
            tr_loss_list.append(tr_loss)
            tr_acc_list.append(tr_acc)
            if X_ts is not None and y_ts is not None:
                ts_loss, ts_acc = self.evaluate(X_ts, y_ts)
                ts_loss_list.append(ts_loss)
                ts_acc_list.append(ts_acc)

            message = (
                f"Epoch {epoch + 1}, Train Loss {tr_loss:.2f}, Train Accu {tr_acc:.4f}"
            )
            if X_ts is not None and y_ts is not None:
                message += f", Test Accu {ts_acc:.4f}"
            print(message)
            logging.info(message)

        if X_ts is not None and y_ts is not None:
            return tr_loss_list, tr_acc_list, ts_loss_list, ts_acc_list
        return tr_loss_list, tr_acc_list

    def evaluate(self, x_data, y_data):
        loss_list = []
        w = self.w.transpose()
        dist = np.array(
            [np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))]
        )

        result = np.argmax(dist, axis=1)
        accuracy = sum(result == y_data) / float(len(y_data))

        loss_list = [neg_log_loss(dist[i], y_data[i]) for i in range(len(y_data))]
        loss = sum(loss_list)
        return loss, accuracy

    def predict(self, x_data):
        w = self.w.transpose()
        dist = np.array(
            [
                np.squeeze(softmax(np.matmul(x_data[i], w)))
                for i in range(x_data.shape[0])
            ]
        )
        result = np.argmax(dist, axis=1)
        return result

# python logistic.py --n_epoches 15 --batch_size 10 --learning_rate 0.005 --decay 0.75
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoches", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import logging

    logging.basicConfig(
        filename="./logreg_mnist.log",
        filemode="w",
        format="%(message)s",
        level=logging.DEBUG,
    )
    print("loading data...")
    x_tr, y_tr, x_ts, y_ts = load_mnist("./mnist/")
    clf = LogClf(
        n_epoches=args.n_epoches,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        decay=args.decay,
        mu=args.mu,
        seed=42,
        verbose=args.verbose,
    )
    print("training...")
    tr_losses, tr_accs, ts_losses, ts_accs = clf.fit(x_tr, y_tr, x_ts, y_ts)
    _, ts_accuracy = clf.evaluate(x_ts, y_ts)
    print(f"Final Test Accuracy: {ts_accuracy*100:.2f}%")
    plot_log(tr_losses, tr_accs, "evaluate/logistic_train.png")
    plot_log(ts_losses, ts_accs, "evaluate/logistic_test.png")
    plot_confusion(clf.predict(x_ts), y_ts, "evaluate/logistic_confmat.png")
    plot_weights(clf, "evaluate/logistic_weights.png")
    plot_predictions(clf, x_ts, y_ts, "evaluate/logistic_predictions.png")
