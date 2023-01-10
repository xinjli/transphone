import pip
import importlib

def import_with_auto_install(package, package_name):
    try:
        return importlib.import_module(package)
    except ImportError:
        pip.main(['install', package_name])
    return importlib.import_module(package)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]