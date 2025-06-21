from model import * 
from solver import *
from config import *

config = basic_config()
schedule_table = config.schedule_table
time_costs = config.time_costs
A = Train(schedule_table[0], time_costs[0], config)
test_node = Node(2,52,False)

path = hir_dijkstra(A, 2, default_value_func)
print(path.check_node(test_node))
