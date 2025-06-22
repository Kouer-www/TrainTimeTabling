from model import * 
from AUG_model import *
from config import *

config = basic_config()
T = AUGTimetable(config)
T.optim()
