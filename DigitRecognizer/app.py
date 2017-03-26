import network
import mnist
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

WEIGHTS_FILE = "./net_weights.txt"
BIASES_FILE = "./net_biases.txt"
SUBMISSION_FILE = "./submission.csv"

MASK_NUMBERS = [np.array([[.0]] * i + [[1.]] + [[.0]] * (9 - i))
                for i in range(10)]

def read_biases():
    result = []
    actual = []
    acum = 0
    with open(BIASES_FILE, "r") as f:
        sizes = map(int, f.readline().split(","))
        for size in sizes:
            for line, value in enumerate(f):
                if line < size - acum:
                    actual.append(float(value))
                else:
                    result.append(actual)
                    actual = [float(value)]
                    acum = 1
                    break
            else:
                result.append(actual)
    return result

def read_weights():
    result = []
    actual = []
    acum = 0
    with open(WEIGHTS_FILE, "r") as f:
        sizes = map(int, f.readline().split(","))
        for n_inputs, n_neurons in zip(sizes[:-1], sizes[1:]):
            for line, value in enumerate(f):
                if line < n_neurons - acum:
                    actual.append(map(float, value.split(",")))
                else:
                    result.append(actual)
                    actual = [map(float, value.split(","))]
                    acum = 1
                    break
            else:
                result.append(actual)
    return result

def run_submission():
    df_test = pd.read_csv("test.csv")
    neurons = [df_test.shape[1], 100, 10]
    net = network.Network(neurons, biases=read_biases(), weights=read_weights())
    with open(SUBMISSION_FILE, "w") as f:
        f.write("ImageId,Label\n")
        for line, row in enumerate(df_test.values):
            f.write("{},{}\n".format(line + 1, net.predict(row)))

def train():

    def extract_data(df):
        X = df.ix[:, "pixel0":]
        X = X.astype(float)
        Y = df["label"]
        dictionary = list(X.columns.values)
        return X, Y, dictionary

    df = pd.read_csv("train.csv")
    X, y, dictionary = extract_data(df)
    # Scale X, if not scaled, results are apalling
    scaler = MinMaxScaler()
    X = pd.DataFrame(data=scaler.fit_transform(X), columns=dictionary)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.3, random_state=42, stratify=y_test)

    training_data = [(np.array([[float(xtt)] for xtt in np.asarray(xt[1])]), MASK_NUMBERS[y_train[xt[0]]])
		    for xt in X_train.iterrows()]
    test_data = [(np.array([[float(xtt)] for xtt in np.asarray(xt[1])]), y_test[xt[0]])
		    for xt in X_test.iterrows()]

    # 20, 5, 2.0 -> Epoch 7: 8256 / 8820
    # 30, 10, 3.0 -> Epoch 24: 8296 / 8820
    # 40, 10, 3.0 -> Epoch 35: 8320 / 8820

    epoch = 35
    min_batch = 10
    eta = 3.0
    print "Runing network with epoch={0} min_batch={1} and eta={2}.".format(
       epoch, min_batch, eta)

    net = network.Network([len(dictionary), 100, 10])
    net.SGD(training_data, epoch, min_batch, eta, test_data=test_data)

    # save values into txt files
    with open(WEIGHTS_FILE,'w') as f:
        f.write(', '.join([str(x) for x in net.sizes]) + "\n")
        for layer in net.weights:
            np.savetxt(f, layer, delimiter=',')

    with open(BIASES_FILE,'w') as f:
        f.write(', '.join(str(len(layer)) for layer in net.biases) + "\n")
        for layer in net.biases:
            np.savetxt(f, layer, delimiter=',')

def train_from_nielsen():
    training_data, validation_data, test_data = mnist.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        # train()
        pass
    elif sys.argv[1] == 'submit':
        run_submission()
    else:
        train_from_nielsen()
