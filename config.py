from dataclasses import dataclass

# Toy Model Settings
schedule_table =[[1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]]

time_cost = [[9, 18, 12, 7, 7, 8],
    [9, 18, 12, 7, 7, 8],
    [9, 18, 12, 7, 7, 8],
    [9, 18, 12, 7, 7, 8],
    [9, 18, 12, 7, 7, 8],
    [9, 18, 12, 7, 7, 8],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10],
    [10, 20, 14, 8, 8, 10]]

dist_list = [0, 50, 100, 170, 210, 250, 300]

class basic_config():
    num_states: int = 7 # number of stations
    num_trains: int = 16 # number of trains
    max_time: int = 160 # max time 
    min_stay: int = 2 # min wait time at each station
    max_stay: int = 15 # max wait time at each station
    N: int = 2 # half of the confliction time 
    miu: float = 0.01 # learning rate
    max_steps: float = 50 # max optimization steps
    schedule_table: list = schedule_table # schedule table for trains
    time_costs: list = time_cost # time cost of between stations of trains
    dist_list: list = dist_list # distance list of the stations
    threshold: float = 0.1 # Threshold for SPP
    sigma: float = 1.5 # sigma used in AUG model
    

