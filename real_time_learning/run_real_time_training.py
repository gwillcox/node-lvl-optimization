"""
Runs a real-time learning experiment by by:
0) loading a network
1) initializing the network's IO
2) running the simulation until a breakpoint is reached
"""

import IO_Utils

import numpy as np
import agents
import network
import matplotlib.pyplot as plt

np.random.seed(3)

n_trials_max = 1000000

# TODO: separate networks and datasets into separate files. Leave this one just for running it.
# TODO: develop and test global simple random search
# TODO: build in nonlinear agents--RELU or random forests?
#
# Defines a dataset of 100 points following a simple plane
# Initializes
io_utils = IO_Utils.utils()
n_inputs = io_utils.n_inputs
n_processing = 2
n_outputs = io_utils.n_outputs

# Initializes a simple network of size 1, with linear agents.
# net = network.TelephoneNetwork(n_agents=3)
net = network.FullyConnectedNetwork(n_inputs=n_inputs, n_agents=n_processing, n_outputs=n_outputs)

# Runs the simulation by training the network for the 100 timesteps
errors = []
predictions = []
desired_outcomes = []
for i in range(n_trials_max):
    if i % 10 == 0:
        print("Percent Finished: " + str(i/n_trials_max*100))
    # Gets the last inputs
    last_inputs = io_utils.get_inputs()

    if len(last_inputs) != n_inputs:
        raise(AssertionError, "Unexpected Input length")

    # Feeds forward information through the network
    last_outputs = net.predict(last_inputs)

    # Takes action with the network's prediction
    io_utils.apply_outputs(last_outputs)

    # Logs the true value of the prediction.
    desired_outcome = np.sign(last_inputs[0])

    # Defines the error function for this experiment
    error = -abs(desired_outcome - last_outputs[-1])

    # Logs this error in the network
    net.log(error)

    # Logs variables of interest!
    predictions.append(last_outputs[-1])
    errors.append(error)
    desired_outcomes.append(desired_outcome)

# [plt.plot(np.asarray(errors)[:,i], label="agent: x_"+str(i+1)) for i in range(len(net.agents))]
plt.plot(errors)
plt.grid()
plt.xlabel("Trial Number")
plt.ylabel("Network Error")
plt.legend()
plt.show()

# [plt.scatter(np.asarray(predictions)[:,i],dataset_y) for i in range(len(net.agents))]
# plt.xlabel("Network Prediction")
# plt.ylabel("True Value")
# plt.show()

plt.scatter(np.asarray(predictions)[-100:],desired_outcomes[-100:])
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

