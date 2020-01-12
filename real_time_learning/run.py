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

# TODO: separate networks and datasets into separate files. Leave this one just for running it.
# TODO: develop and test global simple random search
# TODO: build in nonlinear agents--RELU or random forests?
# TODO: 

# Defines a dataset of 100 points following a simple plane
n_inputs = 32
dataset_x = np.random.rand(n_trials,n_inputs)
weights = np.random.rand(n_inputs)
biases = np.random.rand(n_inputs)-0.5
dataset_y = np.sum(dataset_x*weights+biases,axis=1)/2 + 0.5

# Initializes a simple network of size 1, with linear agents.
# net = network.TelephoneNetwork(n_agents=3)
net = network.MergeNetwork(n_inputs=n_inputs, n_agents=2, n_outputs=1)

# Runs the simulation by training the network for the 100 timesteps
errors = []
predictions = []
for i in range(n_trials):
    if i % n_trials/10 == 0 :
        print("Percent Finished: " + str(i/n_trials*100))
    # Feeds forward information through the network
    prediction = net.predict(dataset_x[i])

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

# Plots all agent utilities
[plt.plot(agent.utility_history, label="agent: x_"+str(agent.id+1)) for agent in net.agents]
plt.grid()
plt.xlabel("Trial Number")
plt.ylabel("Agent utility")
plt.legend()
plt.show()

# Plots the last agent utility  agent utilities
plt.plot(net.agents[-1].utility_history[2:])
plt.grid()
plt.xlabel("Trial Number")
plt.ylabel("Network Error (last node utility)")
plt.legend()
plt.show()

# TODO: Why do some of the agents have positive utility over time?




# TODO: Why do we find the same time to convergence and the same variance in predictions when the number of nodes
# TODO: changes dramatically? Maybe this requires a node-level theory appraoch.
# TODO: Maybe run these results for many random seeds to measure statistically relevant quantities?

print("END")
