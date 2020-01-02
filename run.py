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

np.random.seed(3)

n_trials = 100000

# TODO: separate networks and datasets.
# TODO: Create a 2D-input datatset and a new network to run it.
# Defines a dataset of 100 points following a simple line
dataset_x = np.random.rand(n_trials)
dataset_y = dataset_x/2 + 0.5

# Initializes a simple network of size 1, with linear agents.
net = network.TelephoneNetwork(n_agents=3)

# Runs the simulation by training the network for the 100 timesteps
errors = []
predictions = []
for i in range(n_trials):
    # Feeds forward information through the network
    prediction = net.predict([dataset_x[i]])

    # Logs the true value of the prediction.
    net.log(-abs(dataset_y[i]-prediction[-1]))

    # Logs variables of interest!
    predictions.append(prediction)
    errors.append(prediction-dataset_y[i])

[plt.plot(np.asarray(errors)[:,i], label="agent: x_"+str(i+1)) for i in range(len(net.agents))]
plt.grid()
plt.xlabel("Trial Number")
plt.ylabel("Network Error")
plt.legend()
plt.show()

# [plt.scatter(np.asarray(predictions)[:,i],dataset_y) for i in range(len(net.agents))]
# plt.xlabel("Network Prediction")
# plt.ylabel("True Value")
# plt.show()

plt.scatter(np.asarray(predictions)[-100:,-1],dataset_y[-100:])
plt.xlabel("Last 100 Predictions")
plt.ylabel("True Values")
plt.show()

[plt.plot(agent.utility_history, label="agent: x_"+str(agent.id+1)) for agent in net.agents]
plt.grid()
plt.xlabel("Trial Number")
plt.ylabel("Last node utility")
plt.legend()
plt.show()

# TODO: Why do we find the same time to convergence and the same variance in predictions when the number of nodes
# TODO: changes dramatically? Maybe this requires a node-level theory appraoch.
# TODO: Maybe run these results for many random seeds to measure statistically relevant quantities?

print("END")
