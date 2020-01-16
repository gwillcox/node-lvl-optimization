"""
Defines a set of experiments, and formats them for use in this algorithm!
Experiments must have:
    -A function that stores the network
    -A function that gets the next input and desired output
    -A function that grades the output from the network
"""
import numpy as np
import gym

class LinearFit_1_1:
    """
    Defines a simple linear problem y=x/2+.5
    """

    def __init__(self, n_inputs=1, n_outputs=1):
        self.n_inputs = 1
        self.n_outputs = 1
        self.net = None
        self.next_desired_output = 0
        return

    def get_next_trial(self):
        """
        Gets the x data for the next trial
        :return: Returns a list of the inputs to the network
        """
        x_next = np.random.rand()
        self.next_desired_output = x_next/2 + 0.5
        return [x_next]

    def grade_output(self, output):
        """
        Grades the output of the network as the difference between the
        """
        return -abs(output[0]-self.next_desired_output)


class LinearFit_n_1:
    """
    A problem of fitting a linear transformation from n inputs to 1 output
    """

    def __init__(self, n_inputs=8, n_outputs=None):
        self.n_inputs = n_inputs
        self.n_outputs = 1
        self.next_desired_output = 0

        # Defines the weights that will be used to generate output values for this network
        self.weights = np.random.rand(self.n_inputs)-0.5
        self.biases = np.random.rand(self.n_inputs)-0.5
        return

    def get_next_trial(self):
        """
        Gets the x data for the next trial and stores the desired output
        :return:
        """
        x_next = np.random.rand(self.n_inputs)-0.5
        self.next_desired_output = np.sum(x_next * self.weights + self.biases)
        return x_next

    def grade_output(self, output):
        """
        Grades the output of the network as the difference between the
        """
        return -abs(output - self.next_desired_output)


class LinearFit_n_n:
    """
    A problem of fitting a linear transformation from n inputs to 1 output
    """

    def __init__(self, n_inputs=8, n_outputs=3):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.next_desired_output = 0

        # Defines the weights that will be used to generate output values for this network
        self.weights = np.random.rand(self.n_inputs,n_outputs)-0.5
        self.biases = np.random.rand(self.n_inputs,n_outputs)-0.5
        return

    def get_next_trial(self):
        """
        Gets the x data for the next trial and stores the desired output
        :return:
        """
        x_next = np.random.rand(self.n_inputs)-0.5
        self.next_desired_output = np.sum(x_next.reshape(-1,1) * self.weights + self.biases, axis=0)
        return x_next

    def grade_output(self, output):
        """
        Grades the output of the network
        """
        return -sum(abs(output - self.next_desired_output))



class SqrtFit_n_n(LinearFit_n_n):
    """
    A problem of fitting a sqrt x linear transformation from n inputs to 1 output
    """

    def get_next_trial(self):
        """
        Gets the x data for the next trial and stores the desired output
        :return:
        """
        x_next = np.random.rand(self.n_inputs)-0.5
        self.next_desired_output = np.sum(np.sqrt(x_next.reshape(-1,1)) * self.weights + self.biases, axis=0)
        return x_next

# TODO: add the real-time human input experiment

# TODO: incorporating GYM experiments requires a major restructure:
# TODO: action - > reward - > observation -> action

# class GymCartPole:
#     """
#     Loads the CartPole
#     :return:
#     """
#     def __init__(self, n_inputs=8, n_outputs=1):
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#
#         # Sets up the environment
#         self.env = gym.make("CartPole-v1")
#         self.observation = self.env.reset()
#         self.reward = 0
#         self.done = False
#         return
#
#     def get_next_trial(self):
#         """
#         Gets the x data for the next trial and stores the desired output
#         :return:
#         """
#         self.env.render()
#         action = self.env.action_space.sample()  # your agent here (this takes random actions)
#         observation, reward, done, info = self.env.step(action)
#         return observation
#
#     def grade_output(self, output):
#         """
#         Grades the output of the network
#         """
#         return -sum(abs(output - self.next_desired_output))
