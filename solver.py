from model import *
from copy import deepcopy
import numpy as np

    
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
    print(final_path_list[max_index])
    return final_path_list[max_index]

class LagrangianRelaxation:
    
    def __init__(self,):
        pass 
    