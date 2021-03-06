"""
Runs a simulation of node-level optimization by:
0) Defining a dataset
1) Loading a network
2) Running the simulation for a period of time
"""
import numpy as np
import network
import experiments
import datetime
import visualizations
import error_propagation

np.random.seed(2)

n_trials = 100000
n_seeds = 10
n_inputs = 5
n_agents = 5
n_outputs = 3

# Defines a dataset of 100 points following a simple plane
experiment_type = experiments.LinearFit_n_n
error_function = error_propagation.propagate_global_error
net_type = network.LayerNetworkReLU

# Tracking variables for this simulation
errors = np.empty((n_seeds,n_trials))
predictions = np.empty((n_seeds,n_trials))

# Iterates over the seeds and
for seed in range(n_seeds):

    experiment = experiment_type(n_inputs=n_inputs, n_outputs=n_outputs)

    # Creates a new network
    net = net_type(error_function=error_function, n_inputs=experiment.n_inputs, n_agents=n_agents, n_outputs=experiment.n_outputs)

    # Iterates over the trials for a given seed
    for trial in range(n_trials):
        if trial % 1000 == 0 :
            print("Percent Finished: " + str((seed*n_trials+trial)/(n_trials*n_seeds)))
        # Gets the next problem
        inputs = experiment.get_next_trial()

        # Feeds forward information through the network
        outputs = net.predict(inputs)

        # Gets the graded result
        error = experiment.grade_output(outputs[-experiment.n_outputs:])

        # Logs the true value of the prediction.
        net.log(error)

        # Tracks variables of interest!
        predictions[seed,trial] = outputs[-1]
        errors[seed,trial] = error

# Saves the data for future reference
print("Saving...")
trial_info = {'net': str(net_type).split('.')[1][:-2],
              'error': error_function.__name__,
              'experiment':  str(experiment_type).split('.')[1][:-2]}
filename = 'data/'+trial_info['net'] + "_" + trial_info['error']+"_"+trial_info['experiment']+'_'+datetime.datetime.now().strftime("%H%M")
np.save(filename+"_predictions",predictions)
np.save(filename+"_errors",errors)
np.save(filename+"_trial_info",trial_info)

visualizations.analyze(filename)
visualizations.plot_utility_history(net, trial_info)
visualizations.visualize_connections(net)

print("END")
