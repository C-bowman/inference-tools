try:
    from importlib.metadata import version, PackageNotFoundError, packages_distributions
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError, packages_distributions
try:
    # module is "inference", but package is "inference-tools"
    __version__ = version(packages_distributions()[__name__][0])
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]
