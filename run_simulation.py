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
import matplotlib.pyplot as plt

# TODO: Refactor this into a number of different files: AGENTS, NETWORK, and SIMULATION.

# def load_iris():
#     # Loads the IRIS dataset
#     iris = datasets.load_iris()
#     x = iris.data
#     y = iris.target
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#     return x_train, x_test, y_train, y_test
#
# x_train, x_test, y_train, y_test = load_iris()


# Defines the nodes and connection architecture
n_nodes = 50

def _initialize_connections():
    """
    Sets up the connections to be random feed-forward.
    """
    # Creates a random upper triangular matrix
    connections = np.zeros(shape=(n_nodes,n_nodes))
    for i in range(n_nodes):
        connections[i,i+1:n_nodes] = np.random.rand(n_nodes-i-1)-0.5

    return connections

def _initialize_nodes():
    """
    Initializes n nodes with the parameters we care about
    """
    nodes = {}
    nodes['connections'] = _initialize_connections()
    nodes['signals'] = np.zeros((n_nodes,1))
    nodes['profit'] = np.zeros((n_nodes,1))

    return nodes

def _predict(inputs):
    """
    Propagates information from the first to the final node.
    """
    # First, resets all signals.
    nodes['signals'] = inputs
    for n in range(n_nodes):
        # Makes this node a ReLU unit
        nodes['signals'][n,0] = abs(nodes['signals'][n,0]-0.1)

        # Sends this signal to other nodes
        nodes['signals'] += nodes['signals'][n,0]*nodes['connections'][n].reshape(-1,1)

    return nodes['signals']

def _calculate_importances():
    return abs(nodes['connections'])

def _calculate_error_squared(label):
    """
    Calculates the difference between the label and the last node's voltage.
    Assumes the last node's voltage is all we care about when recording the error to the label.
    """
    return np.square(label-nodes['signals'][-1,0])

def _calculate_error_linear(label):
    """
    Calculates the difference between the label and the last node's voltage.
    Assumes the last node's voltage is all we care about when recording the error to the label.
    """
    return label-nodes['signals'][-1,0]

def _calculate_importance_change(recent_reward):
    """
    For a given model, d
    """

def _update_importances(label):

    reward = np.zeros(n_nodes)
    importance_change = np.zeros(n_nodes)

    # Iterates over nodes
    for n in range(n_nodes-1,-1):
        if n==n_nodes-1:
            reward[-1] = _calculate_error_linear(label)
        else:
            print('none')
        # TODO: Calculates the "reward" given to that agent at this timestep as the difference between the desired and the expected outputs for the output.
        # TODO: Calculates the conection update for a given node as the signal that node sent (node signal times connection weight) times the positivity of the effect that signal had on this node.
        # TODO: Draw this as a diagram to make the process clear.

        _calculate_importance_change()
        importance_change[]


    return None

nodes = _initialize_nodes()

sample_input = np.zeros(n_nodes).reshape(-1,1)
prediction = _predict(sample_input)

sample_input_2 = np.ones(n_nodes).reshape(-1,1)
prediction_2 = _predict(sample_input_2)

sample_input_3 = np.zeros(n_nodes).reshape(-1,1)
sample_input_3[0] = 1
prediction_3 = _predict(sample_input_3)

# TODO: Calculate the importance of each incoming connection
# TODO: Craft a reward function that propagates importance and updates connection strengths.

print("END")
