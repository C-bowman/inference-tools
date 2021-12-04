from warnings import warn

warn(
    """\n
    [ Deprecation warning ]
    >> The 'gp_tools' module has been renamed to 'gp' - import
    >> from inference.gp instead. Support for importing from 
    >> 'gp_tools' will be removed in a future update.
    """
)

from inference.gp import *
