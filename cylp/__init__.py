import os
from os.path import realpath, join
currentDir = os.path.dirname(realpath(__file__))
__version__ = open(join(currentDir, 'VERSION')).read().strip()
