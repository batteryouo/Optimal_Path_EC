__version__ = "1.0.0"
__author__ = "Battery"
__email__ = "oyinggaio@gmail.com"
__license__ = "MIT License"

from .ec import objective
from .ec.shape import Line
from .ec.func import *


__all__ = ["objective", "shape", "func"]