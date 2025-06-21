from config import *
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt 
"""
    We first need a definition of nodes. A node is just a pair (state_label, time, if_out)
"""
    
class Node:
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

ERROR_NODE = Node(-1,-1,False)
START_NODE = Node(0, 0, False)

class Edge:
    node_start: Node
    node_end: Node
    value: float
    use: int = 0

    def __init__(self, node_start, node_end):
        self.node_start = node_start
        self.node_end = node_end
        self.use = 0

class Path:
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
            print("Node height exceeded!")
            return False 
        else: 
            node1 = self.node_list[2 * node.state_label - 1]
            node2 = node1
            if 2 * node.state_label < self.length:
                node2 = self.node_list[2 * node.state_label]
            if node1 == node or node2 == node:
                return True
            return False
    
    def __repr__(self):
        result = "Path: "
        for i in range(self.length):
            result += f"\n{self.node_list[i]}--->"
        result += f"\nLength: {self.length}, Height: {self.height}, Total value {self.value}"
        return result

class Train:

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
        self.init_node = Node(0,0,True)
        self.path = Path(self.init_node)

    def _get_delta_in(self, node: Node):
        # given a node, return the delta-in of this node as a list. !!!!!!!!!!!!! IF WE NEED TO CONSIDER THE STARTING AND DESTINATION STATION?
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
    
    def _get_delta_out(self, node: Node):
        # given a node, return the delta-out of this node as a list. !!!!!!!!!!!!! IF WE NEED TO CONSIDER THE STARTING AND DESTINATION STATION?
        try:
            if node.if_out == False:
                assert node.state_label != 0
                assert node.state_label != self.num_states-1
                assert node.time <= self.config.max_time - self.config.min_stay - 1
                return [Node(node.state_label, node.time+i, False) for i in range(self.config.min_stay, 1 + min(self.config.max_stay, self.config.max_time-1-node.time), 1)]

            elif node.if_out == True:
                assert node.state_label != self.num_states - 1
                time_elapsed = self.time_cost[node.state_label]
                assert time_elapsed <= self.config.max_time - node.time - 1
                return [Node(node.start_label+1, node.time+time_elapsed, False)]
            
        except:
            raise ValueError("--------------------Invalid node structure!!!--------------------")
        
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
        delta_in = self._get_delta_in(node)
        if delta_in[0] == ERROR_NODE:
            return 0
        z = 0
        for tmp_node in delta_in:
            z += 1 if self.path.check_node(tmp_node) == True else 0
        return z
    
    def _get_figure(self):
        result_x = np.zeros(self.path.length)
        result_y = np.zeros(self.path.length)
        for i in range(self.path.length):
            result_x[i] = self.config.dist_list[self.path.node_list[i].state_label]
            result_y[i] = self.path.node_list[i].time
        return result_x, result_y

def default_value_func(node_start: Node, node_end: Node, train: Train):
    assert node_end.time >= node_start.time
    if node_start == START_NODE: 
        return 0
    return -node_end.time+node_start.time

def hir_dijkstra(train: Train, max_states: int, value_func):
    final_path_list = []
    init_path = Path(START_NODE)
    for idx in range(train.max_time):
        init_node = Node(0, idx, True)
        path_list = [init_path.dup_extend(init_node, value_func, train)]
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
            tmp_path_list = []
            for time in range(start_node.time, end_node.time+1, 1):
                tmp_node = Node(start_node.state_label, time, True)
                time_elapsed, tmp_min_node, tmp_max_node = train._backward_two_stage_out(tmp_node)
                tmp_value = []

                for tmp_time in range(tmp_min_node.time, tmp_max_node.time+1):
                    if tmp_time < min_time:
                        continue
                    if tmp_time > max_time:
                        break
                    tmp_path = path_list[tmp_time - min_time]
                    mid_node = Node(tmp_node.state_label, tmp_path.end_node.time+time_elapsed, False)
                    # print(tmp_path.end_node, mid_node, tmp_node)
                    tmp_value.append(tmp_path.value + value_func(tmp_path.end_node, mid_node, train) + value_func(mid_node, tmp_node, train))
                tmp_value = np.array(tmp_value)
                max_index = np.argmax(tmp_value)
                # print(time, max_index, tmp_min_node, max_index+tmp_min_node.time+time_elapsed)
                node_list = [Node(tmp_node.state_label, max_index+max(min_time,tmp_min_node.time)+time_elapsed, False), tmp_node]
                tmp_path_list.append(path_list[max_index+max(tmp_min_node.time-min_time, 0)].dup_extend_multi(node_list, value_func, train))
            path_list = tmp_path_list
        max_time = train._get_delta_in(Node(init_node.state_label + max_states, train.max_time-1, False))[-1].time
        max_index = min(len(path_list), len(path_list)+max_time-path_list[-1].end_node.time) 
        if max_index == 0:
            break
        path_list = path_list[:max_index]
        path_value = np.array([tmp_path.value for tmp_path in path_list])
        max_index = np.argmax(path_value)
        result_path = path_list[max_index]
        result_path = result_path.dup_extend(Node(init_node.state_label + max_states, result_path.end_node.time+train.time_cost[-1], False), value_func, train)
        final_path_list.append(result_path)
    path_value = np.array([tmp_path.value for tmp_path in final_path_list])
    max_index = np.argmax(path_value)
    # print(final_path_list[max_index])
    return final_path_list[max_index]

class Timetable:

    def __init__(self, config: basic_config):
        self.config = config
        self.num_states = config.num_states
        self.num_trains = config.num_trains
        self.max_time = config.max_time
        self.N = config.N
        self.max_steps = config.max_steps
        self.miu = config.miu
        self.value_func = default_value_func
        self._init_nodes()
        self._init_trains()

    def _init_nodes(self):
        self.nodes_lam = []
        for i in range(self.num_states):
            self.nodes_lam.append([[0, 0] for time in range(self.max_time)])

    def _init_trains(self):
        self.trains = []
        for i in range(self.num_trains):
            self.trains.append(Train(self.config.schedule_table[i], self.config.time_costs[i], self.config))
    
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

    def _update_value_func(self):
        def new_value_func(node_start: Node, node_end: Node, train: Train):
            result = default_value_func(node_start, node_end, train)
            node_list = train._get_delta_in(node_end)
            if node_list[0] == ERROR_NODE:
                return result
            for node in node_list:
                if node == node_start:
                    lam = self.nodes_lam[node_end.state_label][node_end.time][0 if node_end.if_out == False else 1]
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
        for train in self.trains:
            train.path = hir_dijkstra(train, self.num_states-1, self.value_func)
        self._update_value_func()
        self._update_lam()

    def draw_figure(self, output_path = "result.png"):
        print("--------------------Drawing figure ... --------------------")
        plt.figure()
        plt.ylim(0, self.config.dist_list[-1])
        plt.xlim(0, self.max_time)
        plt.ylabel("Distance")
        plt.xlabel("Time")
        for train in self.trains:
            tmp_x, tmp_y = train._get_figure()
            print(tmp_y)
            plt.plot(tmp_y, tmp_x, "r--")  
        plt.savefig(output_path)
        plt.close()

    def optim(self):
        with tqdm(total=self.max_steps) as pbar:
            for _ in range(self.max_steps):
                self._optim_loop()
                pbar.update(1)
        self.draw_figure()
