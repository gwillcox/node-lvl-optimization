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


class TelephoneNetwork:
    """
    Creates a network of agents strung together in 1d, like a game of telephone.
    """

    def __init__(self, n_agents=1):
        """
        Initializes the network, specifying each agent and which agents connect to one another
        """
        # TODO: Let's build in the definitions of an input node. That'll allow us to experiment with mutliple inputs,
        # TODO: and test whether the errors we're seeing are exclusive to line-line networks.
        self.agents = [agents.LinearSRS(agent_id=i, n_connections_in=2) for i in range(n_agents)]
        self.connections = [[i,i+1] for i in range(n_agents-1)]

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
        # Sets up the outcome tracker--which records the error / change in utility for each agent.
        agent_outcome = np.zeros(len(self.agents))
        agent_outcome[-1] += outcome  # Sets the last agent's error to the outcome.

        # Iterates over agents, from last to first,
        for i in range(len(self.agents)-1,-1,-1):
            agent = self.agents[i]

            # Record the outcome for this agent, and associate it with the agent's last inputs.
            agent.store_data(agent_outcome[agent.id])

            # Update the method this agent uses for signalling based on this new data
            agent_importance_change = agent.update()

            # Iterates over the connections and propagates the signal reward.
            # Assumes that the first connections found will be the first signals reported
            connection_num = 0
            for connection in self.connections:
                if connection[1] == agent.id:
                    # TODO: This currently only works for agents that send a single signal. Fix this to index better (below).
                    agent_outcome[connection[0]] += agent_importance_change[connection_num+1]
                    # TODO: Figure out how better to index connections. Sending them forward is easy, because the same
                    # TODO: signal is sent to all other nodes, but sending utility backwards requires a better structure
                    connection_num += 1
        return
