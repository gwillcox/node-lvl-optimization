"""
Defines a set of data saving functions that gather data that agents report.
"""
import numpy as np
import datetime

def save_numpy_data_to_file(data, filename):
    """
    Saves numpy data to a file
    """
    if type(data) != type(np.array([])):
        raise(AssertionError,"Bad Type of Data. Should be Numpy Array")
    np.save('data/save_states/'+filename, data)

def save_network_data(network, filename=None):
    """
    Saves a network's data.
    :param network:
    :return:
    """
    # Gathers data from the network. Data must be in a JSON format
    dataset = np.array(network.save_data())

    # Saves this data into a file with either the title provided or a timestamp
    if filename is None:
        save_numpy_data_to_file(dataset, 'save_sattes/'+str(datetime.time))
    else:
        save_numpy_data_to_file(dataset, 'save_states/'+filename)

    print("Saved data to filename: ")

