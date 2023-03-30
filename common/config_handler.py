import json
from types import SimpleNamespace
import yaml
import os

CONFIG_FILE = os.environ.get('CONFIG_FILE')

class Config():
    """
    Retrieves the info from config.json file and it \
    converts it into a class for easy access for all components \
    in the system.

    This is a Singleton class.
    """
    __shared_instance = None

    def __new__(cls):
        if cls.__shared_instance is None:
            cls.__shared_instance = super().__new__(cls)
            cls.get = cls.load_config()
        
        return cls.__shared_instance  

    @classmethod
    def load_config(cls):

        with open(CONFIG_FILE) as config_file:
            data_map = yaml.safe_load(config_file)

        return json.loads(json.dumps(data_map), object_hook = lambda d : SimpleNamespace(**d))
        