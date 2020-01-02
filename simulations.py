"""
Contains a set of simulations that can be easily run.
Simulations:
    -Define the dataset
    -Define the network structure

"""
import numpy as np
import agents
import network
import matplotlib.pyplot as plt

def linear_1d_sim(n_trials):

    """
    Defines a simple 1inear 1d simulation.
    """
    # Defines a dataset of 100 points following a simple line
    dataset_x = np.random.rand(n_trials)
    dataset_y = dataset_x/2 + 0.5

    # Initializes a simple network of size 1, with linear agents.
    net = network.Network()

    # Runs the simulation by training the network for the 100 timesteps
    errors = []
    predictions = []
    for i in range(1000):
        # Feeds forward information through the network
        prediction = net.predict([dataset_x[i]])

        # Logs the true value of the prediction.
        net.log(dataset_y[i])

        # Logs variables of interest!
        predictions.append(prediction)
        errors.append(prediction-dataset_y[i])

    plt.plot(errors)
    plt.grid()
    plt.show()

    # TODO: Why is this network producing steady-state values that are incorrect?
    [plt.scatter(np.asarray(predictions)[:,i],dataset_y) for i in range(len(net.agents))]
    plt.show()

