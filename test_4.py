from model import * 
from config import *

config = basic_config()
T = Timetable(config)
T.optim()
