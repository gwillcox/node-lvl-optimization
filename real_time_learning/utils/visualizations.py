import numpy as np
import matplotlib.pyplot as plt

def analyze(f):
    # Loads the relevant data
    errors = np.load(f+"_errors.npy")
    predictions = np.load(f+"_predictions.npy")
    trial_info = np.load(f+"_trial_info.npy", allow_pickle=True).item()

    visualize_error_profile(predictions[:,-100:], errors[:,-100:], trial_info)

    plot_bootstrapped_performance(np.arange(len(errors[0])),errors, trial_info,
                                  xlabel="Trial Number", ylabel="Trial Error")

    plot_bootstrapped_performance_trial(np.arange(len(errors[0])),errors, trial_info,
                                  xlabel="Trial Number", ylabel="Average Update Error")


def plot_bootstrapped_performance(x, y, trial_info, xlabel=None, ylabel=None):
    """
    Creates a bootstrapped line graph showing the average, 50%, and 90% Confidence Intervals
    """
    sorted_y = np.sort(y,axis=0)
    median = np.median(y, axis=0)
    n_samples = len(y)
    CI_50 = sorted_y[[int(n_samples/4),int(3*n_samples/4)-1],:]
    CI_90 = sorted_y[[int(n_samples/10), int(9*n_samples/10)-1],:]

    plt.fill_between(x,CI_90[0],CI_90[1], label="90% CI", color="orange")
    plt.fill_between(x,CI_50[0],CI_50[1], label="50% CI", color="purple")
    plt.plot(x,median, label="Median", color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.title(trial_info["net"]+" | "+trial_info["error"]+" | "+trial_info["experiment"])
    plt.show()


def plot_bootstrapped_performance_trial(x, y, trial_info, xlabel=None, ylabel=None):
    """
    Creates a bootstrapped line graph showing the average, 50%, and 90% Confidence Intervals
    """
    # Converts x and y to steps that measure the performance over set updates
    n_steps_per_trial = 50
    x_in_steps = np.arange(0,len(x),n_steps_per_trial)
    y_in_steps = []
    for i in x_in_steps:
        y_in_steps.append(np.mean(y[:,np.arange(i,i+50)], axis=1))
    y_in_steps = np.array(y_in_steps).transpose()

    # Calculates plottable statistics
    sorted_y = np.sort(y_in_steps,axis=0)
    median = np.median(y_in_steps, axis=0)
    n_samples = len(y_in_steps)
    CI_50 = sorted_y[[int(n_samples/4),int(3*n_samples/4)-1],:]
    CI_90 = sorted_y[[int(n_samples/10), int(9*n_samples/10)-1],:]

    # Plots the thing!
    plt.fill_between(x_in_steps,CI_90[0],CI_90[1], label="90% CI", color="orange")
    plt.fill_between(x_in_steps,CI_50[0],CI_50[1], label="50% CI", color="purple")
    plt.plot(x_in_steps,median, label="Median", color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.title(trial_info["net"]+" | "+trial_info["error"]+" | "+trial_info["experiment"])
    plt.show()


def plot_utility_history(net, trial_info):
    """
    Plots the agent utilities for the last network recorded
    """
    [plt.plot(agent.utility_history, label="agent: x_" + str(agent.id + 1)) for agent in net.agents]
    plt.grid()
    plt.xlabel("Trial Number")
    plt.ylabel("Agent utility")
    plt.legend()
    plt.title(trial_info["net"]+" | "+trial_info["error"]+" | "+trial_info["experiment"])
    plt.show()


def visualize_connections(net):
    """
    Creates a matrix-visualization of this network's connections
    """
    n_agents = len(net.agents)
    array = np.zeros((n_agents, n_agents))
    for item in net.connections:
        array[item[0], item[1]] = 1

    plt.imshow(array)
    plt.ylabel("Connecting FROM")
    plt.xlabel("Connecting TO")
    plt.show()


def visualize_error_profile(predictions, errors, trial_info):
    """
    Creates a scatterplot of the error of a network
    """
    # Selects the last 100 data
    plt.scatter(predictions, errors)
    plt.xlabel("Network Predictions")
    plt.ylabel("Network Errors")
    plt.title(trial_info["net"]+" | "+trial_info["error"]+" | "+trial_info["experiment"])
    plt.show()