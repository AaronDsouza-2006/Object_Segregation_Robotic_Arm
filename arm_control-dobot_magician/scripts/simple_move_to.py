import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pydobotplus'))

from pydobotplus import dobotplus as dobot

# Example usage of the pydobot library
device = dobot.Dobot(port='COM3')
device.move_to(100, 0, -30)
device.close()
