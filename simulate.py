"""
Runs a simulation of node-level optimization by:
0) Defining a dataset
1) Loading a network
2) Running the simulation for a period of time
"""
import numpy as np
import agents
import network
import matplotlib.pyplot as plt

np.random.seed(1)

# Defines a dataset of 100 points following a simple line
dataset_x = np.random.rand(1000)
dataset_y = dataset_x/2 + 0.5

# Initializes a simple network of size 1, with linear agents.
net = network.Network()

# Runs the simulation by training the network for the 100 timesteps
errors = []
predictions = []
for i in range(1000):
    prediction = net.predict(dataset_x[i])

    net.log(dataset_x[i], dataset_y[i])
    # TODO: find a better way to log this. At the moment, we're sending the raw data... but we really want nodes to track their own input + signals.

    # Logs variables of interest!
    predictions.append(prediction)
    errors.append(prediction-dataset_y[i])

plt.plot(errors)
plt.grid()
plt.show()

plt.scatter(predictions,dataset_y)
plt.show()

plt.scatter(predictions[-100:],dataset_y[-100:])
plt.xlabel("predictions")
plt.ylabel("dataset y")
plt.show()

print("END")
