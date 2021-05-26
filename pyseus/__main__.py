"""Defines ways to start PySeus.

Methods
-------

**load(arg)** - Startup function and console entry point.
"""

# argv gives all input arguments that are used while calling script in CL as a list
from sys import argv
import numpy

from .core import PySeus



# standard value of arg if not defined is None
def load(arg=None):
    """Start Pyseus and load *arg* (can be a path or data array)."""

    app = PySeus()
    if len(argv) > 1 and arg is None:
        arg = argv[1]
    if isinstance(arg, str):
        app.load_file(arg)
    elif isinstance(arg, (numpy.ndarray, list)):
        app.load_data(arg)

    return app.show()


if __name__ == "__main__":
    load()
