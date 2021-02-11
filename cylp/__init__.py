import os
from os.path import realpath, join
currentDir = os.path.dirname(realpath(__file__))
with open(join(currentDir, 'VERSION')) as f:
    __version__ = f.read().strip()
