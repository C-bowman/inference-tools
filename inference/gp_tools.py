
from warnings import warn

warn(
    """
    [ Depreciation warning ]
    >> The 'gp_tools' module has been renamed to 'gp'. Support for
    >> importing from 'gp_tools' will be removed in a future update.
    """
)

from inference.gp import *
