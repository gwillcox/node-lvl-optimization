#####
# Creates a set of real-time agents that make independent decisions about when to update.
# All Real Agents have:
#   INIT
#   Signal Method
#   Update Signalling Method
#   Calculate Signal Importance
#   Update Signal Importance
#####
import scipy.stats
import numpy as np

class Agent:
    """
    A simple linear agent class. Can be used for output agents, and is extended below.
    """

    def __init__(self, agent_id, n_connections_in):
        """
        Creates a linear agent. This agent has:
         -n unit input connections
         -a 100-long history of inputs and outputs
        """
        # Records this agent's ID
        self.id = agent_id

        # initializes the input history with a fit of 1... just to start the model off.
        self.connections = np.ones(n_connections_in)

        # Tracks the recent connections to measure change in connection importance
        self.last_importance = np.zeros(self.connections.shape)

        # Tracking variables for this agent's history
        self.last_inputs = np.zeros(n_connections_in)
        self.input_history = [np.zeros(n_connections_in), np.ones(n_connections_in)/n_connections_in]
        self.utility_history = []
        self.history_length = 1000

    def signal(self, input):
        """
        Processes a set of inputs into a single signal output. Records this last input
        """
        self.last_inputs = input
        return np.sum(self.connections*np.asarray(input))

    def store_data(self, utility):
        """
        Keeps track of the recent inputs + errors.
        """
        self.input_history.append(np.asarray(self.last_inputs))
        self.utility_history.append(utility)

        if len(self.input_history) > self.history_length:
            self.input_history.pop(0)
            self.utility_history.pop(0)  # assumes the histories are of the same length

    def update(self):
        """
        Updates the signalling method, then returns the change in signal importance that each incoming node presents.
        """
        # Updates the signalling method.
        self.update_signal_method()

        # Calculates the new signal importance of the updated signalling method, and returns it.
        new_signal_importance = self.calculate_signal_importance()
        signal_importance_change = new_signal_importance - self.last_importance
        self.last_importance = new_signal_importance.copy()

        return signal_importance_change

    def update_signal_method(self):
        """
        Updates the signalling method--in this case, the agent is minimizing the error to the desired output by changing
         the connections that activate the linear unit.
        Then, returns the change in signal importance that each incoming node presents.
        """
        # Calculates the new signalling method
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.asarray(self.input_history).reshape(-1),
                                                                             np.asarray(self.utility_history).reshape(-1))
        self.connections = np.array(slope)

    def calculate_signal_importance(self):
        """
        Calculates the signal importance as the absolute value of the connection strengths
        """
        return abs(self.connections)


class InputAgent(Agent):
    """
    This agent works to send linear inputs into the network. It doesn't ever update, and returns a NONE value for
    signal importance.
    """
    def __init__(self, agent_id):
        super().__init__(agent_id, n_connections_in=0)

    def signal(self, input):
        """Returns the input values as a float"""
        return input[1]

    def store_data(self, utility):
        return

    def update(self):
        return None


class LinearSRS(Agent):
    """
    This agent uses a random exploration rule to maximize it's perceived utility.
    """

    def __init__(self, agent_id, n_connections_in):
        super().__init__(agent_id, n_connections_in)

        # logs the random searches and time to a new search
        self.time_to_update = 200
        self.time_since_update = self.time_to_update+1
        self.recent_update = np.zeros(self.connections.shape)
        self.expected_utility = -np.inf

    def signal(self, input):
        """
        Processes a set of inputs into a single signal output. Records this last input
        """
        self.last_inputs = input
        return np.sum(self.connections*np.asarray(input))

    def store_data(self, utility):
        """
        Keeps track of the recent inputs + errors.
        """
        self.input_history.append(np.asarray(self.last_inputs))
        self.utility_history.append(utility)

    def update(self):
        """
        Updates the signalling method, then returns the change in signal importance that each incoming node presents.
        """
        # Updates the signalling method.
        self.update_signal_method()

        # Calculates the new signal importance of the updated signalling method, and returns it.
        signal_importance = self.calculate_signal_importance()

        return signal_importance

    def update_signal_method(self):
        """
        Checks whether we've spent enough time evaluating the new connections for utility. If so, updates!
        """
        # Increment time, and check if we've spent enough time waiting to evaluate the effectiveness of this update
        self.time_since_update += 1
        if self.time_to_update < self.time_since_update:

            average_utility = np.mean(self.utility_history[-self.time_to_update:])

            # If the new utility is worse than expected, then undo the changes
            if average_utility < self.expected_utility:
                self.connections -= self.recent_update

            # Makes a new update!
            # TODO: Does changing this update fn change the steady-state error?
            self.recent_update = (np.random.rand(len(self.connections))-0.5)/20
            self.connections += self.recent_update

            # Resets the expected utility
            self.expected_utility = average_utility

            # Resets the time since update
            self.time_since_update = 0

    def calculate_signal_importance(self):
        """
        Calculates the signal importance as: (signal_sent x connection_str) for each incoming connection x utility
        """
        signal_importance = self.input_history[-1] * self.connections * self.utility_history[-1]
        return signal_importance


class ReLUSRS(LinearSRS):
    """
    This agent is the same as Linear SRS but rectifies the signal also
    """

    def signal(self, input):
        """
        Processes a set of inputs into a single signal output using a RELU function
        """
        self.last_inputs = input

        signal = np.sum(self.connections*np.asarray(input))
        signal = max(signal, 0)
        return signal
