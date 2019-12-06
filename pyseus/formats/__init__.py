"""Formats model different data sources.

All formats extend the *BaseFormat* class, guaranteeing basic
functionality of checking files and data, loading files and data and
extracting frames and metadata.
"""

from .base import BaseFormat, LoadError
from .raw import Raw
from .numpy import NumPy
from .h5 import H5
from .dicom import DICOM
from .nifti import NIfTI
