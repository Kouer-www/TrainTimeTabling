from model import * 
from solver import *
from config import *

config = basic_config()
T = Timetable(config)
T.optim()
