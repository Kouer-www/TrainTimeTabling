from config import *
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import time

import matplotlib.pyplot as plt 

COLOR_SET = ["r-", "b-", "g-", "c-", "m-", "y-", "k-"]
    
class Node:
    """
        A node is just a pair (state_label, time, if_out)
    """
    state_label: int
    time: int # from 0 to max_time-1
    if_out: bool
    
    def __init__(self, state_label, time, if_out):
        self.state_label = state_label
        self.time = time
        self.if_out = if_out
        self.lam = 0

    def __repr__(self):
        tmp = "in" if self.if_out == False else "out"
        return f"Station: {self.state_label}, Time: {self.time}, Status: {tmp}"
    
    def __eq__(self, node):
        return (self.state_label == node.state_label) & (self.time == node.time) & (self.if_out == node.if_out)

# some special Nodes
ERROR_NODE = Node(-1,-1,False)
START_NODE = Node(0, 0, False)


class Path:
    """
        A path is a list of node, with some metrics.
    """
    node_list: list 
    length: int
    height: int 
    value: float
    end_node: Node
    def __init__(self, init_node):
        self.node_list = [init_node]
        self.length = 1
        self.height = 0
        self.value = 0
        self.end_node = init_node
    
    def _extend(self, node, value_func, train):
        self.value += value_func(self.end_node, node, train)
        self.length += 1
        self.node_list.append(node)
        self.end_node = node
        self.height = self.end_node.state_label
    
    def _extend_multi(self, node_list, value_func, train):
        self.value += value_func(self.end_node, node_list[0], train)
        for i in range(len(node_list)-1):
            self.value += value_func(node_list[i], node_list[i+1], train)
        self.length += len(node_list)
        self.node_list += node_list
        self.end_node = node_list[len(node_list)-1]
        self.height = self.end_node.state_label

    def dup_extend(self, new_node, value_func, train):
        Path = deepcopy(self)
        Path._extend(new_node, value_func, train)
        return Path
    
    def dup_extend_multi(self, node_list, value_func, train):
        Path = deepcopy(self)
        Path._extend_multi(node_list, value_func, train)
        return Path
    
    def check_node(self, node):
        if node.state_label > self.height:
            return False 
        else: 
            node1 = self.node_list[2 * node.state_label]
            if node1 == node:
                return True
            if 2 * node.state_label+1 < self.length:
                node2 = self.node_list[2 * node.state_label+1]
                if node2 == node:
                    return True
            return False
    
    def check_edge(self, node_start, node_end):
        return self.check_node(node_start) == True & self.check_node(node_end) == True
    
    def __repr__(self):
        result = "Path: "
        for i in range(self.length):
            result += f"\n{self.node_list[i]}--->"
        result += f"\nLength: {self.length}, Height: {self.height}, Total value {self.value}"
        return result
    
    def get_value(self, value_func, train):
        value = 0 
        if self.length == 1:
            return 0
        for i in range(self.length-1):  
            value += value_func(self.node_list[i], self.node_list[i+1], train)
        return value
    
    def __eq__(self, path):
        if self.length != path.length:
            return False 
        for i in range(self.length):
            if self.node_list[i] != path.node_list[i]:
                return False
        return True

# some special Paths
ERROR_PATH = Path(ERROR_NODE)


class PATH_LIST:
    """
        A customized data structure to index a path using the last node of the path
    """
    path_list: list 
    node_list: list

    def __init__(self):
        self.path_list = []
        self.node_list = []

    def append(self, path):
        self.path_list.append(path)
        self.node_list.append(path.end_node)
    
    def check(self, node):
        for i, tmp_node in enumerate(self.node_list):
            if node == tmp_node:
                return self.path_list[i], i
        return ERROR_PATH, -1
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        return self.path_list[index]

class Train:
    """
        A train is a combination of the structures above. The most important things are:
            1. self.path: The path the train currently has
            2. self.SPP_path: Similar to self.path, but is used in SPP phase.
            3. _get_delta_in: Return the possible in-degree of a node.
    """
    def __init__(self, scheduler: list, time_cost: list, config: basic_config):
        assert len(scheduler) == config.num_states
        assert len(time_cost) == config.num_states - 1
        self.config = config
        self.num_states = config.num_states
        self.scheduler = scheduler
        self.time_cost = time_cost
        self.min_stay = config.min_stay
        self.max_stay = config.max_stay 
        self.max_time = config.max_time
        self.N = config.N
        self.path = ERROR_PATH
        self.SPP_path = ERROR_PATH

    def _get_delta_in(self, node: Node):
        # Given a node, return the possible nodes which can reach it.
        if node.if_out == False:
            # implies that it's an in-node 
            if node.state_label == 0:
                return [ERROR_NODE]
            time_elapsed = self.time_cost[node.state_label-1]
            if node.time < time_elapsed:
                return [ERROR_NODE]
            return [Node(node.state_label-1, node.time-time_elapsed, True)]
        
        elif node.if_out == True:
            if self.scheduler[node.state_label] == 0:
                return [Node(node.state_label, node.time, True)]
            # implies that it's an out-node
            if node.state_label == 0:
                return [START_NODE]
            assert node.state_label != self.num_states-1
            if node.time < self.config.min_stay:
                return [ERROR_NODE]
            return [Node(node.state_label, node.time-i, False) for i in range(self.config.min_stay, 1 + min(self.config.max_stay, node.time), 1)]
        
    def _forward_two_stage_out(self, node: Node):
        assert node.if_out == True
        assert node.state_label != self.num_states-1
        time_elapsed = self.time_cost[node.state_label]
        if self.scheduler[node.state_label+1] == 0 and ((time_elapsed + node.time) < self.max_time):
            return time_elapsed, Node(node.state_label+1,node.time+time_elapsed, True), Node(node.state_label+1,node.time+time_elapsed, True)
        min_time = time_elapsed + self.min_stay
        if min_time + node.time >= self.max_time:  
            return time_elapsed, ERROR_NODE, ERROR_NODE
        max_time = min(self.max_time - 1 - node.time, time_elapsed + self.max_stay)
        return time_elapsed, Node(node.state_label+1,node.time+min_time,True), Node(node.state_label+1,node.time+max_time,True)
    
    def _backward_two_stage_out(self, node: Node):
        assert node.if_out == True
        assert node.state_label != 0
        time_elapsed = self.time_cost[node.state_label-1]
        if self.scheduler[node.state_label] == 0:
            return time_elapsed, Node(node.state_label-1,node.time-time_elapsed, True), Node(node.state_label-1,node.time-time_elapsed, True)
        min_time = time_elapsed + self.min_stay
        if min_time > node.time:
            return time_elapsed, ERROR_NODE, ERROR_NODE
        max_time = min(node.time, time_elapsed + self.max_stay)
        return time_elapsed, Node(node.state_label-1,node.time-max_time, True), Node(node.state_label-1, node.time-min_time, True)

    def z(self, node: Node):
        if self.path == ERROR_PATH:
            return 0
        if self.path.check_node(node) == False:
            return 0
        delta_in = self._get_delta_in(node)
        if delta_in[0] == ERROR_NODE:
            return 0
        z = 0
        for tmp_node in delta_in:
            if self.path.check_node(tmp_node) == True:
                z += 1
        return z
    
    def _get_figure(self, SPP_phase = False):
        path = self.path
        if SPP_phase == True:
            path = self.SPP_path
        result_x = np.zeros(path.length)
        result_y = np.zeros(path.length)
        if path == ERROR_PATH:
            return result_x, result_y
        for i in range(path.length):
            result_x[i] = self.config.dist_list[path.node_list[i].state_label]
            result_y[i] = path.node_list[i].time
        return result_x, result_y
    
    def get_value(self, value_func):
        return self.path.get_value(value_func, self)
    
def index_node_table(node_table, node):
    try:
        return node_table[node.state_label][node.time][1 if node.if_out == True else 0]
    except: 
        return None

def check_usable_table(node: Node, usable_table):
    return index_node_table(usable_table, node) == 1

def default_value_func(node_start: Node, node_end: Node, train: Train):
    assert node_end.time >= node_start.time
    if node_start == START_NODE: 
        return -node_end.time
    return -node_end.time+node_start.time

"""
    Main function of optimization
"""
def hir_dijkstra(train: Train, max_states: int, value_func, usable_table):
    final_path_list = PATH_LIST()
    init_path = Path(START_NODE)
    for idx in range(train.max_time):
        if usable_table[0][idx][1] == 0:
            continue
        init_node = Node(0, idx, True)
        path_list = PATH_LIST()
        path_list.append(init_path.dup_extend(init_node, value_func, train))
        start_node = init_node
        end_node = init_node
        for id in range(max_states-1):
            min_time = start_node.time
            max_time = end_node.time 
            _, start_node, _ = train._forward_two_stage_out(start_node)
            _, _, end_node = train._forward_two_stage_out(end_node)
            if start_node == ERROR_NODE:
                break
            elif end_node == ERROR_NODE:
                end_node = Node(start_node.state_label, train.max_time-1, True)
            tmp_path_list = PATH_LIST()
            for time in range(start_node.time, end_node.time+1, 1):
                tmp_node = Node(start_node.state_label, time, True)
                if check_usable_table(tmp_node, usable_table) == False:
                    continue
                time_elapsed, tmp_min_node, tmp_max_node = train._backward_two_stage_out(tmp_node)
                tmp_value = []
                index_list = []
                for tmp_time in range(tmp_min_node.time, tmp_max_node.time+1):
                    if tmp_time < min_time:
                        continue
                    if tmp_time > max_time:
                        break
                    tmp_path, index = path_list.check(Node(tmp_node.state_label-1, tmp_time, True))
                    if tmp_path == ERROR_PATH:
                        continue
                    mid_node = Node(tmp_node.state_label, tmp_path.end_node.time+time_elapsed, False)
                    if check_usable_table(mid_node, usable_table) == False:
                        continue
                    # print(tmp_path.end_node, mid_node, tmp_node)
                    tmp_value.append(tmp_path.value + value_func(tmp_path.end_node, mid_node, train) + value_func(mid_node, tmp_node, train))
                    index_list.append(index)
                if len(tmp_value) == 0:
                    continue
                tmp_value = np.array(tmp_value)
                max_index = np.argmax(tmp_value)
                tmp_index = index_list[max_index]
                tmp_path1 = path_list[tmp_index]
                # print(time, max_index, tmp_min_node, max_index+tmp_min_node.time+time_elapsed)
                node_list = [Node(tmp_node.state_label, tmp_path1.end_node.time+time_elapsed, False), tmp_node]
                tmp_path_list.append(tmp_path1.dup_extend_multi(node_list, value_func, train))
            path_list = tmp_path_list
            if len(path_list) == 0:
                break
        if len(path_list) == 0:
            continue
        index=[]
        for i in range(len(path_list)):
            tmp_time = path_list[i].node_list[-1].time + train.time_cost[-1]
            if tmp_time < train.max_time and usable_table[-1][tmp_time][0] == 1:
                index.append(i)
        if len(index) == 0:
            break
        path_list = [path_list[i] for i in index]
        path_value = np.array([tmp_path.value for tmp_path in path_list])
        max_index = np.argmax(path_value)
        result_path = path_list[max_index]
        result_path = result_path.dup_extend(Node(init_node.state_label + max_states, result_path.end_node.time+train.time_cost[-1], False), value_func, train)
        final_path_list.append(result_path) 
    if len(final_path_list) == 0:
        return ERROR_PATH
    path_value = np.array([tmp_path.value for tmp_path in final_path_list.path_list])
    max_index = np.argmax(path_value)
    # print(final_path_list[max_index])
    return final_path_list[max_index]

class Timetable:
    """
        Basic timetable solver using Lagrange relaxation in Problem 4.
    """
    def __init__(self, config: basic_config):
        self.config = config
        self.num_states = config.num_states
        self.num_trains = config.num_trains
        self.max_time = config.max_time
        self.N = config.N
        self.max_steps = config.max_steps
        self.miu = config.miu
        self.threshold = config.threshold
        self.value_func = default_value_func
        self._init_nodes()
        self._init_trains()
        self._init_draw()

    def _init_nodes(self):
        # initialize the lambda table and usable table of nodes.
        self.nodes_lam = np.ones((self.num_states, self.max_time, 2), dtype=np.float32)
        self.nodes_usable = np.ones((self.num_states, self.max_time, 2), dtype=np.int8)
        """
        self.nodes_lam = []
        self.nodes_usable = []
        for i in range(self.num_states):
            self.nodes_lam.append([[1, 1] for time in range(self.max_time)])
            self.nodes_usable.append([[1, 1] for _ in range(self.max_time)])
        """

    def _init_trains(self):
        # initialize trains
        self.trains = []
        for i in range(self.num_trains):
            self.trains.append(Train(self.config.schedule_table[i], self.config.time_costs[i], self.config))
    
    def _init_draw(self):
        # initialize records for drawing
        self.vio_record = []
        self.upper_record = []
        self.lower_record = []
        self.time_record = []

    def _reset_usable_table(self):
        # reset the usable table
        self.nodes_usable = np.ones((self.num_states, self.max_time, 2), dtype=np.int8)
        """
        for i in range(self.num_states):
            for j in range(self.max_time):
                self.nodes_usable[i][j] = [1, 1]
        """

    def y(self, node: Node):
        y = 0
        for train in self.trains:
            y += train.z(node)
        return y 
    
    def vio(self, node: Node):
        min_v = max(node.time - self.N, 0)
        max_v = min(node.time + self.N, self.max_time-1)
        vio = 0
        for time in range(min_v, max_v+1):
            tmp_node = Node(node.state_label, time, node.if_out)
            vio += self.y(tmp_node)
        return vio - 1
    
    def total_vio(self):
        # compute the total violation
        total = 0
        for i in range(self.num_states):
            for j in range(self.max_time):
                total += max(0, self.vio(Node(i, j, False)))
                if i != self.num_states-1: # We ignore the end point
                    total += max(0, self.vio(Node(i, j, True)))
        return total
    
    def upper_bound(self):
        # compute the upper bound
        total = 0
        for train in self.trains:
            total += train.get_value(default_value_func)
        return total
    
    def lower_bound(self):
        # compute the lower bound
        total = 0
        for train in self.trains:
            total += train.get_value(self.value_func)
        total += self.total_lam()
        return total
    
    def total_lam(self):
        # compute the total lambda
        total = 0
        for i in range(self.num_states):
            for j in range(self.max_time):
                total += self.nodes_lam[i][j][0]
                if i != self.num_states-1:
                    total += self.nodes_lam[i][j][1]
        return total
    
    def total_train(self, SPP_phase=False):
        # compute how many trains are used 
        total = 16
        for train in self.trains:
            if SPP_phase == True:
                if train.SPP_path == ERROR_PATH:
                    total -= 1
                continue
            if train.path == ERROR_PATH:
                total -= 1
        return total
    
    def total_time(self):
        total = 0
        for i in self.time_record:
            total += i
        return total 
    
    def recent_time(self):
        return self.time_record[-1]

    def _update_value_func(self):
        # Add penalty to the value function using Lagrange Relaxation
        def new_value_func(node_start: Node, node_end: Node, train: Train):
            result = default_value_func(node_start, node_end, train)
            node_list = train._get_delta_in(node_end)
            if node_list[0] == ERROR_NODE:
                return result
            for node in node_list:
                if node == node_start:
                    lam = index_node_table(self.nodes_lam, node_end)
                    result -= lam * (min(node_end.time+self.N, self.max_time-1) - max(node.time-self.N, 0))
                    break
            return result
        self.value_func = new_value_func

    def _update_lam(self):
        for i in range(self.num_states):
            for j in range(self.max_time):
                self.nodes_lam[i][j][0] = max(0, self.nodes_lam[i][j][0] + self.miu * self.vio(Node(i, j, False)))
                if i != self.num_states-1:
                    self.nodes_lam[i][j][1] = max(0, self.nodes_lam[i][j][1] + self.miu * self.vio(Node(i, j, True)))

    def _optim_loop(self):
        """
        1. hir_dijkstra solve the result 
        2. return the path to each train 
        3. update the result of z and y 
        4. update lambda and value_func
        """
        self._reset_usable_table()
        for i, train in enumerate(self.trains):
            print(f"----------Mini Batch ({i+1}/{self.num_trains})----------")
            train.path = hir_dijkstra(train, self.num_states-1, self.value_func, self.nodes_usable)
            self._update_lam()
            self._update_value_func()
            self.vio_record.append(self.total_vio())
            self.upper_record.append(self.upper_bound())
            self.lower_record.append(self.lower_bound())

    def draw_timetable(self, output_path = "result/timetable.png", SPP_phase = False):
        print("--------------------Drawing timetable ... --------------------")
        count = 0
        plt.figure()
        plt.ylim(0, self.config.dist_list[-1])
        plt.xlim(0, self.max_time)
        plt.ylabel("Distance")
        plt.xlabel("Time")
        for train in self.trains:
            tmp_x, tmp_y = train._get_figure(SPP_phase)
            print(tmp_y)
            if tmp_x[-1] != 0: 
                plt.plot(tmp_y, tmp_x, COLOR_SET[count%len(COLOR_SET)])  
            count += 1
        plt.savefig(output_path)
        plt.close()

    def draw_vio(self, output_path = "result/violation.png"):
        print("--------------------Drawing violation ... --------------------")
        y = np.array(self.vio_record)
        plt.figure()
        plt.xlim(0, y.shape[0]-1)
        plt.ylim(0, 1.1 * np.max(y))
        plt.ylabel("Violation")
        plt.xlabel("Steps")
        plt.plot(np.arange(y.shape[0]), y, "r-")
        plt.savefig(output_path)
        plt.close()
    
    def draw_object(self, output_path = "result/objective.png"):
        print("--------------------Drawing object ... --------------------")
        y1 = np.array(self.upper_record)
        y2 = np.array(self.lower_record)
        plt.figure()
        plt.xlim(0, y1.shape[0]-1)
        plt.ylim(0, 1.1 * min(np.min(y1), np.min(y2)))
        plt.ylabel("Function value")
        plt.xlabel("Steps")
        plt.plot(np.arange(y1.shape[0]), y1, "r-")
        plt.plot(np.arange(y1.shape[0]), y2, "b-")
        plt.savefig(output_path)
        plt.close()

    def draw_figure(self):
        self.draw_timetable()
        self.draw_vio()
        self.draw_object()
    
    def metric(self):
        metric = {}
        metric["total_vio"] = self.total_vio()
        metric["upper_bound"] = self.upper_bound()
        metric["lower_bound"] = self.lower_bound()
        metric["total_lam"] = self.total_lam()
        metric["total_train"] = self.total_train()
        metric["total_time"] = self.total_time()
        metric["recent_time"] = self.recent_time()
        return metric

    def optim(self):
        for i in range(self.max_steps):
            time1 = time.time()
            self._optim_loop()
            time2 = time.time()
            self.time_record.append(time2-time1)
            text = f"--------------------Process ({i+1}/{self.max_steps})--------------------"
            metrics = self.metric()
            for name, value in metrics.items():
                text += f"\n{name}: {value}"
            print(text)
            if abs(self.upper_record[-1] - self.lower_record[-1]) < self.threshold * abs(self.upper_record[-1]):
                print("--------------------Trying SPP ... --------------------")
                
                result = self.SPP()
                print(f"SPP result {result} Trains: {self.total_train(True)}")
                if result == True:
                    print("Successfully arranged!!!")
                    break 
        self.draw_figure()
        

    def _update_usable_table(self, path):
        if path == ERROR_PATH:
            return 
        for node in path.node_list:
            if node == START_NODE:
                continue
            min_v = max(node.time - 2*self.N, 0)
            max_v = min(node.time + 2*self.N, self.max_time-1)
            for time in range(min_v, max_v + 1):
                self.nodes_usable[node.state_label][time][1 if node.if_out == True else 0] = 0

    def SPP(self):
        self._reset_usable_table()
        priority_list = np.zeros(self.num_trains)
        for i, train in enumerate(self.trains):
            priority_list[i] = train.get_value(self.value_func)
        index_list = np.argsort(priority_list)
        for id in index_list:
            train = self.trains[id]
            train.SPP_path = hir_dijkstra(train, self.num_states-1, default_value_func, self.nodes_usable)
            self._update_usable_table(train.SPP_path)
        self.draw_timetable("result/SPP_result.png", True)
        return self.total_train(True) == self.num_trains

        
