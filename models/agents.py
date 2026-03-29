import numpy as np

class Prosumer:
    def __init__(self, agent_id, bus_index, params=None):
        self.agent_id = agent_id
        self.bus_index = bus_index # 连接在电网的哪个节点上
        
        # 1. 电池参数 (来自论文 Section V-Simulation)
        self.q_max = 15.0      # 电池容量 kWh
        self.soc = 0.5 * self.q_max # 初始电量 (State of Charge)
        self.alpha = 0.99      # 漏电率
        self.beta = 0.9        # 充电效率
        self.s_max = 0.7 * self.q_max # 最大充放电功率
        
        # 2. 负荷参数
        self.e_min = 0.0
        self.e_max = 5.0 # 假设的最大功率
        
        # 3. 存储该代理的决策序列 (用于 Receding Horizon)
        self.u_history = [] 

    def update_soc(self, s_t):
        """更新电池状态 (对应论文公式 3a)"""
        self.soc = self.alpha * self.soc + self.beta * s_t
        return self.soc