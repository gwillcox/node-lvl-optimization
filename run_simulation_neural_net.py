"""
Runs a simulation of node-level optimization by:
0) Loading the environment
1) Initializing the nodes and connections
2) Running the algo for a period of time
"""
import numpy as np
import sklearn.datasets as datasets
import sklearn.neural_network
import sklearn.model_selection
import sklearn as sk

def load_iris():
    # Loads the IRIS dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


x, y = load_iris()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,5,))

model.fit(x_train,y_train)

y_predictions = model.predict(x_test)

print(y_predictions)

accuracy = np.average(y_test==y_predictions)

print("final accuracy: ", accuracy)
