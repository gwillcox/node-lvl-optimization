"""
This file contains functions that are used to propagate error between nodes in a network.
All error methods require:
    -Outcome
    -Agents
    -Connections

And return the final agent outcomes.
"""
import numpy as np

def propagate_global_error(outcome, agents, connections):
    """
    Sets all agent outcomes to be the global outcome
    """
    agent_outcome = np.ones(len(agents)) * outcome

    # Stores the global outcome
    for agent in agents:
        agent.store_data(outcome)

        # Update each agent's behavior based on this outcome
        agent.update()

    return agent_outcome

def propagate_signal_importance(outcome, agents, connections):
    """
    Calculates and propagates agent-level signal importance
    """

    # Sets up the outcome tracker--which records the error / change in utility for each agent.
    agent_outcome = np.zeros(len(agents))
    agent_outcome[-1] += outcome  # Sets the last agent's error to the outcome.

    # Iterates over agents, from last to first.
    for i in range(len(agents) - 1, -1, -1):
        agent = agents[i]

        # Record the outcome for this agent, and associate it with the agent's last inputs.
        agent.store_data(agent_outcome[agent.id])

        # Update the method this agent uses for signalling based on this new data
        agent_importance_change = agent.update()

        # Iterates over the connections and propagates the signal reward.
        # Assumes that the first connections found will be the first signals reported (i.e. connections are ordered)
        connection_num = 0
        for connection in connections:
            if connection[1] == agent.id:
                agent_outcome[connection[0]] += agent_importance_change[connection_num + 1]
                connection_num += 1

    return agent_outcome