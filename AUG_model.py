from model import *

class AUGTimetable(Timetable):

    def __init__(self, config: basic_config):
        super().__init__(config)
        self._init_nodes_new()
        self.sigma = self.config.sigma
    
    def _init_nodes_new(self):
        self.nodes_s = np.ones((self.num_states, self.max_time, 2), dtype=np.float32)
        self.nodes_vio = np.zeros((self.num_states, self.max_time, 2), dtype=np.float32)
        
    def _update_value_func(self):
        def new_value_func(node_start: Node, node_end: Node, train: Train):
            result = default_value_func(node_start, node_end, train)
            node_list = train._get_delta_in(node_end)
            if node_list[0] == ERROR_NODE:
                return result
            for node in node_list:
                if node == node_start:
                    lam = index_node_table(self.nodes_lam, node_end)
                    result -= lam * (min(node_end.time+self.N, self.max_time-1) - max(node.time-self.N, 0))
                    result -= self.sigma * (1/2 - train.path.check_edge(node_start, node_end))
                    min_v = max(node_end.time - self.N, 0)
                    max_v = min(node_end.time + self.N, self.max_time - 1)
                    for time in range(min_v, max_v+1):
                        a = node_end.state_label
                        b = 1 if node_end.if_out == True else 0
                        result -= self.sigma * (self.nodes_vio[a][time][b] + self.nodes_s[a][time][b])
                    break
            return result
        self.value_func = new_value_func

    def _update_lam(self):
        for i in range(self.num_states):
            for j in range(self.max_time):
                self.nodes_lam[i][j][0] = max(0, self.nodes_lam[i][j][0] + self.miu * (self.nodes_vio[i][j][0] + self.nodes_s[i][j][0]))
                if i != self.num_states-1:
                    self.nodes_lam[i][j][1] = max(0, self.nodes_lam[i][j][1] + self.miu * (self.nodes_vio[i][j][1] + self.nodes_s[i][j][1]))

    def _update_s(self):
        for i in range(self.num_states):
            for j in range(self.max_time):
                self.nodes_s[i][j][0] = max(0, -self.nodes_lam[i][j][0]/self.sigma - self.nodes_vio[i][j][0])
                if i != self.num_states-1:
                    self.nodes_s[i][j][1] = max(0, -self.nodes_lam[i][j][1]/self.sigma - self.nodes_vio[i][j][1])

    def _update_vio(self):
        for i in range(self.num_states):
            for j in range(self.max_time):
                self.nodes_vio[i][j][0] = self.vio(Node(i, j, False))
                if i != self.num_states-1:
                    self.nodes_vio[i][j][1] = self.vio(Node(i, j, True))
    
    def _total_vio_s_penalty(self):
        total = 0
        for i in range(self.num_states):
            for j in range(self.max_time):
                total += self.nodes_lam[i][j][0] * (self.nodes_vio[i][j][0] + self.nodes_s[i][j][0]) + 0.5 * self.sigma * ((self.nodes_vio[i][j][0] + self.nodes_s[i][j][0])**2)
                if i != self.num_states-1: # We ignore the end point
                    total += self.nodes_lam[i][j][1] * (self.nodes_vio[i][j][1] + self.nodes_s[i][j][1]) + 0.5 * self.sigma * ((self.nodes_vio[i][j][1] + self.nodes_s[i][j][1])**2)
        return total
    
    def lower_bound(self, SPP_phase = False):
        return self.upper_bound(SPP_phase) - self._total_vio_s_penalty()

    def _optim_loop(self):
        self._reset_usable_table()
        for i, train in enumerate(self.trains):
            print(f"----------Mini Batch ({i+1}/{self.num_trains})----------")
            train.path = hir_dijkstra(train, self.num_states-1, self.value_func, self.nodes_usable)
            self._update_vio()
            self._update_s()
            self._update_lam()
            self._update_value_func()
            self.vio_record.append(self.total_vio())
            self.upper_record.append(self.upper_bound())
            self.lower_record.append(self.lower_bound())
            