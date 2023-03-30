import logging 
import sys

class Logger():
    """
    Log everything in the system and write everything \
    into a log.txt file.
    This is a Singleton class.
    """
    __shared_instance = None

    def __new__(cls):
        if cls.__shared_instance is None:
            cls.log = cls.get_logger("delay_flight")
            cls.__shared_instance = super().__new__(cls)
        
        return cls.__shared_instance
        
    @classmethod
    def get_logger(cls, name):
        stdout_handler = logging.StreamHandler(stream=sys.stdout)

        logging.basicConfig(
            level=logging.DEBUG, 
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[stdout_handler]
        )

        return logging.getLogger(name)
