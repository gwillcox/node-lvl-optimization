# TODO: Create a CONNECTIONS visualization function that shows the matrix of agent connections
import numpy as np
import matplotlib.pyplot as plt

def analyze(f):
    # Loads the relevant data
    errors = np.load(f+"_errors.npy")
    predictions = np.load(f+"_predictions.npy")
    net = np.load(f+"_net.npy", allow_pickle=True).item()

    plot_bootstrapped_performance(np.arange(len(errors[0])),errors,
                                  xlabel="Trial Number", ylabel="Error")

def visualize_network_connections(net):
    print(net)

def plot_bootstrapped_performance(x, y, xlabel=None, ylabel=None):
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
    plt.show()

def visualize_connections(net):
    n_agents = len(net.agents)
    array = np.zeros((n_agents, n_agents))
    for item in net.connections:
        array[item[0], item[1]] = 1

    plt.imshow(array)
    plt.ylabel("Connecting FROM")
    plt.xlabel("Connecting TO")
    plt.show()


visualize_connections(None)