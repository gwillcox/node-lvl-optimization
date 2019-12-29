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

# TODO: Update this network to handle multiple agents (including backpropagating the signal between agents)

class Network:
    """
    Assumes the simplest network is a 1-node linear regression
    """

    def __init__(self):
        """
        Initializes the network, specifying each agent
        """
        self.agents = [agents.Agent(1)]

    def predict(self, input):
        """
        Since this is a one-agent network, we can simply log the one agent's predictions given the input.
        :param input:
        :return:
        """
        output = []
        for agent in self.agents:
            output.append(agent.signal(input))

        return output

    def log(self, signals, outcome):
        """
        Logs the true expected outcome, and backpropagates the change in utility.
        """
        agent_outcome = np.zeros(len(self.agents))
        agent_outcome[-1] = outcome

        for i in range(len(self.agents)):
            agent = self.agents[-i]
            agent.store_data([signals], outcome)
            agent.update_signal_importance()
            agent.update_signal_method()

        return

    def update(self):
        """
        Loops through all agents and invokes an update function.
        """
        for agent in self.agents:
            agent.update_signal_method()
            agent.update_signal_importance()

        return

