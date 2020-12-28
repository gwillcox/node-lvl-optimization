import pynput
import pyautogui
import numpy as np


class utils:
    """
    Defines utils that interact with the human in real-time
    """

    def __init__(self, inputs=["keyboard"], outputs=["mouse"]):

        self.n_inputs = 0
        self.listeners = {}
        self.next_inputs = {}
        if "keyboard" in inputs:
            self.start_keyboard_listener()

        print("Activated Real-time Inputs: ", self.n_inputs)

        self.n_outputs = 0
        self.controllers = {}
        if "mouse" in outputs:
            self.start_mouse_controller()

        print("Activated Real-time Outputs: ", self.n_outputs)

    def start_mouse_controller(self):
        self.controllers['mouse'] = pynput.mouse.Controller()
        self.n_outputs += 2

    def get_mouse_position(self):
        mouse_position = pyautogui.position()
        return mouse_position

    def move_mouse(self, dx, dy):
        self.controllers['mouse'].move(dx, dy)
        print("Mouse moved to: " + str(self.controllers['mouse'].position))

    def on_key_press(self, key):
        """
        Logs the last key pressed if it's an up or down key
        """
        print(key)
        if key == pynput.keyboard.Key.up:
            self.next_inputs["keyboard"] = 1
        elif key == pynput.keyboard.Key.down:
            self.next_inputs["keyboard"] = -1
        else:
            print("keypress not found: " + str(key))


    def start_keyboard_listener(self):
        """
        Starts a keyboard listening script
        """
        self.listeners['keyboard'] = pynput.keyboard.Listener(
            on_press=self.on_key_press)
        self.listeners['keyboard'].start()
        self.next_inputs['keyboard'] = 0

        # Assumes the keyboard returns only one input
        self.n_inputs += 1

    def get_last_keypress(self):
        """
        A script to return the most recent keypress
        :return:
        """
        # TODO: Why does this error out immediately? Even when I've pressed keys
        # Good resource: https://pypi.org/project/pynput/

        with self.listeners['keyboard'].Events() as events:
            # Gathers the most recent event
            event = events.get(1.0)
            if event is None:
                print("No key pressed in the last second")
            else:
                print("Received event: " + str(event))
                print("Event -1: " + str(events[-1]))

            return events[-1]

    def get_inputs(self):
        """
        Gathers the inputs for the specified network by checking the network's inputs and returning an array with those inputs
        :return:
        """

        # Iterates over the inputs to the network
        inputs = []
        for listener_key in self.listeners.keys():
            if listener_key == 'keyboard':
                inputs.append(self.next_inputs['keyboard'])

        return inputs

    def apply_outputs(self, network_outputs):
        """
        Gathers all outputs from the network and applies them to this computer!
        """
        network_outputs = list(network_outputs)
        print("Received outputs: ", network_outputs)

        # Iterates over all outputs from the network
        for controller in self.controllers.keys():
            if controller == 'mouse':
                assert (network_outputs.__len__() >= 2, "Not enough network outputs!")
                dx = network_outputs.pop()
                dy = network_outputs.pop()
                self.move_mouse(dx, dy)

