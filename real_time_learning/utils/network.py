#####
# Defines the base class of Network for node-level optimization as a 2-node network, and enables other extension classes
# All Networks have:
#   INIT
#   Predict
#   Log
#   Update
#####
import numpy as np
import agents
import error_propagation


class TelephoneNetwork:
    """
    Creates a network of agents strung together in 1d, like a game of telephone.
    """

    def __init__(self, error_function=None, n_agents=1):
        """
        Initializes the network, specifying each agent and which agents connect to one another
        """
        # TODO: Let's build in the definitions of an input node. That'll allow us to experiment with mutliple inputs,
        # TODO: and test whether the errors we're seeing are exclusive to line-line networks.
        self.agents = [agents.LinearSRS(agent_id=i, n_connections_in=2) for i in range(n_agents)]
        self.connections = [[i,i+1] for i in range(n_agents-1)]
        self.error_function = error_function

    def predict(self, input):
        """
        Calculates each agent's response to the input by feeding foward the signals from each agent to the next.
        :return:
        """
        output = []
        agent_inputs = [[1] for agent in self.agents]  # Creates an array of inputs for each agent at a given timestep. Each agent has at least 1 input--a unitary bias term.
        agent_inputs[0].append(input[0])  # For the first node, append the input.

        # Iterates over agents to propagate the signal from each agent to the next
        for i, agent in enumerate(self.agents):
            agent_signal = agent.signal(agent_inputs[i])

            # sends signals to each agent that connects to this one
            for connection in self.connections:
                if connection[0] == i:
                    agent_inputs[connection[1]].append(agent_signal)

            output.append(agent_signal)  # Gathers the last agent's behavior

        return output

    def log(self, outcome):
        """
        Logs the true expected outcome, and backpropagates the change in utility.
        """

        self.error_function(outcome, self.agents, self.connections)
        return


class MergeNetwork(TelephoneNetwork):
    """
    Creates a network that initializes a n agents to handle all inputs, and 1 agent to handle those agents' interplay
    """

    def __init__(self, error_function=None, n_inputs=4, n_agents=2, n_outputs=1):
        """
        Initializes the network, including random connection strengths.
        """
        super().__init__(error_function,n_agents)
        # Defines the connection structure: half of incoming signals connect to one processing agent, half to the other.
        # Both processing agents connect to the one final output node
        split_pt = n_inputs / n_agents
        self.connections = [[i, n_agents] for i in range(n_agents)]
        self.n_inputs = n_inputs
        self.n_agents = n_agents

        #  Creates agents based on this connection structure
        self.agents = [agents.LinearSRS(agent_id=i, n_connections_in=1+np.sum(np.asarray(self.connections)[:,1]==i)) for i in range(n_agents+n_inputs)]
        self.agents.extend([agents.LinearSRS(agent_id=n_inputs+n_agents+i, n_connections_in=1+n_agents) for i in range(n_outputs)])

        # Logs the error function for this network
        self.error_function = error_function


    def predict(self, input):
        """
        Calculates each agent's response to the input by feeding foward the signals from each agent to the next.
        :return:
        """
        output = []
        agent_inputs = [[1] for agent in self.agents]  # Creates an array of inputs for each agent at a given timestep. Each agent has at least 1 input--a unitary bias term.
        [agent_inputs[i].append(input[i]) for i in range(self.n_inputs)]  # For the first node, append the input.

        # Iterates over agents to propagate the signal from each agent to the next
        for i, agent in enumerate(self.agents):
            agent_signal = agent.signal(agent_inputs[i])

            # sends signals to each agent that connects to this one
            for connection in self.connections:
                if connection[0] == i:
                    agent_inputs[connection[1]].append(agent_signal)

            output.append(agent_signal)  # Gathers the last agent's behavior

        return output


class MergeNetworkReLU(MergeNetwork):
    """
    Creates a network that initializes a n agents to handle all inputs, and 1 agent to handle those agents' interplay
    """

    def __init__(self, error_function=error_propagation.propagate_signal_importance, n_inputs=4, n_agents=2, n_outputs=1):
        """
        Initializes the network, including random connection strengths.
        """
        super().__init__(error_function,n_agents)
        # Defines the connection structure: half of incoming signals connect to one processing agent, half to the other.
        # Both processing agents connect to the one final output node
        split_pt = n_inputs / n_agents
        self.connections = [[i, n_agents] for i in range(n_agents)]
        self.n_inputs = n_inputs
        self.n_agents = n_agents

        #  Creates input + processing units that are RELUs and output agents that are Linear
        self.agents = [agents.ReLUSRS(agent_id=i, n_connections_in=1+np.sum(np.asarray(self.connections)[:,1]==i)) for i in range(n_agents+n_inputs)]
        self.agents.extend([agents.LinearSRS(agent_id=n_inputs+n_agents+i, n_connections_in=1+n_agents) for i in range(n_outputs)])


class LayerNetworkReLU(MergeNetwork):
    """
    Creates a network that consists of one "layer", sending signals from input to the processing layer to output
    """

    def __init__(self, error_function=error_propagation.propagate_signal_importance, n_inputs=4, n_agents=2, n_outputs=1):
        """
        Initializes the network.
        """
        super().__init__(error_function,n_agents)
        self.n_inputs = n_inputs
        self.n_agents = n_agents

        # Sets up the input agent connections: all input agents send information to all processing agents
        self.connections = []
        for processing_agent in range(n_agents):
            self.connections.extend([[i, n_inputs+processing_agent] for i in range(n_inputs)])

        for output_agent in range(n_outputs):
            self.connections.extend([[n_inputs+i, n_inputs+n_agents+output_agent] for i in range(n_agents)])

        #  Creates input + processing units that are RELUs and output agents that are Linear
        self.agents = [agents.ReLUSRS(agent_id=i, n_connections_in=1+np.sum(np.asarray(self.connections)[:,1]==i)) for i in range(n_agents+n_inputs)]
        self.agents.extend([agents.LinearSRS(agent_id=n_inputs+n_agents+i, n_connections_in=1+n_agents) for i in range(n_outputs)])


class FullyConnectedNetwork(TelephoneNetwork):

    def __init__(self, error_function=None, n_inputs=5, n_agents=5, n_outputs=1):
        """
        Initializes the network as fully connected: each agent receives input from all inputs and all other previous agents.
        """
        super().__init__(error_function,n_agents)
        # Defines the connection structure: half of incoming signals connect to one processing agent, half to the other.
        # Both processing agents connect to the one final output node
        split_pt = n_inputs / n_agents
        self.connections = [[i,n_inputs+1] for i in range(n_inputs)]
        self.connections.extend([[i, i+1] for i in range(n_inputs,n_inputs+n_agents)])
        self.connections.extend([[n_inputs+n_agents-1, n_inputs+n_agents+n_outputs-1]])
        self.n_inputs = n_inputs
        self.n_agents = n_agents

        #  Creates agents based on this connection structure
        self.agents = [agents.InputAgent(agent_id=i) for i in range(n_inputs)]
        self.agents.extend([agents.LinearSRS(agent_id=n_inputs+i,
                                             n_connections_in=1+(n_inputs-1)*(i==n_inputs)) for i in range(n_agents+n_outputs)])

    def predict(self, input):
        """
        Calculates each agent's response to the input by feeding foward the signals from each agent to the next.
        :return:
        """
        output = []
        agent_inputs = [[1] for agent in self.agents]  # Creates an array of inputs for each agent at a given timestep. Each agent has at least 1 input--a unitary bias term.
        [agent_inputs[i].append(input[i]) for i in range(self.n_inputs)]  # For the first node, append the input.

        # Iterates over agents to propagate the signal from each agent to the next
        for i, agent in enumerate(self.agents):
            agent_signal = agent.signal(agent_inputs[i])

            # sends signals to each agent that connects to this one
            for connection in self.connections:
                if connection[0] == i:
                    agent_inputs[connection[1]].append(agent_signal)

            output.append(agent_signal)  # Gathers the last agent's behavior

        return output