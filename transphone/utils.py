import pip
import importlib

def import_with_auto_install(package, package_name):
    try:
        return importlib.import_module(package)
    except ImportError:
        pip.main(['install', package_name])
    return importlib.import_module(package)