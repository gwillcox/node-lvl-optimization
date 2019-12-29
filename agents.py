#####
# Defines the base class of Agent for node-level optimization as linear, and enables other extension classes.
# All Agents have:
#   INIT
#   Signal Method
#   Update Signalling Method
#   Calculate Signal Importance
#   Update Signal Importance
#####
import sklearn.linear_model
import scipy.stats
import numpy as np

class Agent:
    """
    A simple linear agent class. Can be used for output agents, and is extended below.
    """

    def __init__(self, n_connections_in):
        """
        Creates a linear agent. This agent has:
         -n unit input connections
         -a 100-long history of inputs and outputs
        """
        # initializes the input history with a fit of 1... just to start the model off.
        self.connections = np.ones(n_connections_in)
        self.bias = 0
        self.input_history = [np.zeros(n_connections_in), np.ones(n_connections_in)/n_connections_in]
        self.utility_history = [0, 1]
        self.history_length = 1000

    def signal(self, input):
        """
        Processes a set of inputs into a single signal output
        """
        return np.sum(self.connections*input) + self.bias

    def store_data(self, inputs, utility):
        """
        Keeps track of the recent inputs + errors.
        TODO: Update this to remember 100 data points?
        """
        self.input_history.append(inputs)
        self.utility_history.append(utility)

        if len(self.input_history) > self.history_length:
            self.input_history.pop(0)
            self.utility_history.pop(0) # assumes the histories are of the same length

    def update_signal_method(self):
        """
        Updates the signalling method--in this case, the agent is minimizing the error to the desired output by changing
         the connections that activate the linear unit.
        """
        # TODO: Why does this regression not return a line with slope 0.5 and intercept 0.5?
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.asarray(self.input_history).reshape(-1),
                                                                             np.asarray(self.utility_history).reshape(-1))
        self.connections = slope
        self.bias = intercept

    def calculate_signal_importance(self):
        """
        Calculates the signal importance as the absolute value of the connection strengths
        """
        return abs(self.connections)

    def update_signal_importance(self):
        """
        Signal importance here is equivalent to the connection strength... so we'll just update the signalling method.
        """
        self.update_signal_method()