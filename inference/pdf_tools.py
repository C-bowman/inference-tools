
from warnings import warn

warn(
    """
    [ Depreciation warning ]
    >> The 'pdf_tools' module has been renamed to 'pdf'. Support for
    >> importing from 'pdf_tools' will be removed in a future update.
    """
)

from inference.pdf import *