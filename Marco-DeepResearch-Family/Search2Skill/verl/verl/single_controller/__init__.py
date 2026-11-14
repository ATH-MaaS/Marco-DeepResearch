import os

from . import base
from .base import *

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# Note(TODO): single_controller.__version__ is deprecated
with open(os.path.join(os.path.join(version_folder, os.pardir), "version/version")) as f:
    __version__ = f.read().strip()

__all__ = base.__all__
