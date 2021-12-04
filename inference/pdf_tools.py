from warnings import warn

warn(
    """\n
    [ Deprecation warning ]
    >> The 'pdf_tools' module has been renamed to 'pdf' - import
    >> from inference.pdf instead. Support for importing from 
    >> 'pdf_tools' will be removed in a future update.
    """
)

from inference.pdf import *
