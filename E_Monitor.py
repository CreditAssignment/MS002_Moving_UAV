import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from E_UAV import UAV, UAVType, UavNeighbor, global_min_uav2uav_trans_rate, global_min_uav_energy
from E_GroundUser import GroundUser

base = UAV(uavID=900,
           uav_position=(10000, 10000, 0),
           uav_speed=0,
           uav_type=UAVType.BASE,
           MAX_ENERGY=100000)  # base作为UAV的实例对象只是为了方便，但是base不是UAV


class Monitor:
    def __init__(self):
        self.uav_set: [int, UAV] = {}  # 不包括Base
        self.ground_user_set: [GroundUser] = {}
        self.num_ALL_UAV = 0
        self.num_Active_UAV = 0
        self.num_Damaged_UAV = 0
        self.num_ground_user = 0
        self.uav_net = {}  # UAV网络图结构的邻接矩阵表示
        self.remaining_uav_ids: [int, float] = {}  # 剩余 UAV 的ID及能量
        self.damaged_uav_ids = {}  # 损毁 UAV 的ID及其最后的位置

    def addUAV(self, uav):
        self.uav_set[uav.uavID] = uav
        self.remaining_uav_ids[uav.uavID] = uav.uav_energy
        self.num_ALL_UAV += 1
        self.num_Active_UAV += 1
        self.updateAllAboutUAVnet()

    def addGroundUser(self, groundUser):
        self.num_ground_user += 1
        self.ground_user_set[groundUser.userID] = groundUser

    def deleteUAV(self, uav):
        # del self.uav_set[uav.uavID]
        del self.remaining_uav_ids[uav.uavID]
        self.damaged_uav_ids[uav.uavID] = uav.uav_energy
        self.num_Active_UAV -= 1
        self.num_Damaged_UAV += 1
        self.updateAllAboutUAVnet()

    def deleteGroundUser(self, groundUser):
        del self.ground_user_set[groundUser.userID]
        self.num_ground_user -= 1

    def setUavEnergy(self, uavID, uav_energy):
        if uavID not in self.uav_set:
            print(f"\033[91m UAV {uavID} not exists \033[0m")
        else:
            self.uav_set[uavID].uav_energy = uav_energy
            self.remaining_uav_ids[uavID] = uav_energy
            self.num_Active_UAV = len(self.remaining_uav_ids)
            self.updateAllAboutUAVnet()

    def moveUAV(self, uavID: int, destination: tuple[float, float, float], moving_duration: float):
        self.uav_set[uavID].uavMove(
            moving_duration=moving_duration,
            destination=destination,
        )
        self.updateAllAboutUAVnet()

    def getUavState(self, uavID):
        uav = self.uav_set[uavID]
        print(
            f"\033[96m [UAV STATE] UAV{uav.uavID} Position{uav.uav_position} Energy{uav.uav_energy} Load{uav.uav_load}\033[0m")

    def getUavServingGroundUsersState(self, uavID):
        uav = self.uav_set[uavID]
        for groundUserID in uav.serving_GroundUser_set:
            self.ground_user_set[groundUserID].infoState()

    def updateAllUavActivities(self, min_energy=global_min_uav_energy):
        # UAV的能量大于等于min_energy，则UAV是active的
        self.remaining_uav_ids.clear()
        self.damaged_uav_ids.clear()
        for uav in self.uav_set.values():
            if uav.uav_energy >= min_energy:
                uav.is_active = True
                self.remaining_uav_ids[uav.uavID] = uav.uav_energy
            else:
                uav.is_active = False
                self.damaged_uav_ids[uav.uavID] = uav.uav_position
        self.num_Active_UAV = len(self.remaining_uav_ids)
        self.num_Damaged_UAV = len(self.damaged_uav_ids)

    def updateEachUavNeighborsWithBase(self, min_uav2uav_trans_rate: float = global_min_uav2uav_trans_rate):
        """
        更新每个 UAV 的 uavNeighborSet，同时构建 Monitor 的 UAV 网络图 self.uavNet。

        判定 UAV j 是否可以成为 UAV i 的邻居：
        1. i 和 j 不是同一架 UAV；
        2. 二者距离 < UAV i 的 communicationRadius；
        3. 由 UAV i 调用 get_uav2uavTransRate(j) 得到的通信速率 > min_trans_rate。

        self.uavNet 是邻接表形式：
        {
            uav_i_id: {neighbor_1_id, neighbor_2_id, ...},
            ...
        }
        """

        # 每次更新前，先清空旧邻居信息
        self.uav_net.clear()
        self.uav_net = {uavID: set() for uavID in self.uav_set.keys() if self.uav_set[uavID].is_active}
        self.uav_net[base.uavID] = set()

        for uav in self.uav_set.values():
            uav.uav_neighbor_set.clear()
            uav.num_UAV_neighbor = 0

        # 遍历每一对 UAV
        for src_uav_id, src_uav in self.uav_set.items():
            src_uav.uav2uav_TransRate.clear()  # 清空src_uav到其他uav的传输速率字典
            # base要特殊处理
            dx = base.uav_position[0] - src_uav.uav_position[0]
            dy = base.uav_position[1] - src_uav.uav_position[1]
            dz = base.uav_position[2] - src_uav.uav_position[2]
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            trans_rate = src_uav.getUav2UavTransRate(base.uavID, base.uav_position)
            if distance <= src_uav.communication_radius and trans_rate >= min_uav2uav_trans_rate:
                uav_neighbor = UavNeighbor(
                    uavID=base.uavID,
                    uav_type=base.uav_type,
                    uav_position=base.uav_position,
                    uav_energy=base.uav_energy,
                    uav_load=base.uav_load,
                    uav_speed=base.uav_speed
                )

                src_uav.appendUAVNeighbor(uav_neighbor)
                self.uav_net[src_uav_id].add(base.uavID)
                self.uav_net[base.uavID].add(src_uav_id)

            for dst_uav_id, dst_uav in self.uav_set.items():

                # UAV 不能把自己作为邻居
                if src_uav_id == dst_uav_id:
                    continue
                # dst必须是active的
                if not self.uav_set[dst_uav_id].is_active:
                    continue

                dx = dst_uav.uav_position[0] - src_uav.uav_position[0]
                dy = dst_uav.uav_position[1] - src_uav.uav_position[1]
                dz = dst_uav.uav_position[2] - src_uav.uav_position[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 条件 1：距离必须小于当前 UAV 的通信半径
                if distance > src_uav.communication_radius:
                    continue

                # 条件 2：通信速率必须大于 60 bps
                trans_rate = src_uav.getUav2UavTransRate(dst_uav_id, dst_uav.uav_position)
                if trans_rate < min_uav2uav_trans_rate:
                    continue

                # 构造邻居信息
                uav_neighbor = UavNeighbor(
                    uavID=dst_uav.uavID,
                    uav_type=dst_uav.uav_type,
                    uav_position=dst_uav.uav_position,
                    uav_energy=dst_uav.uav_energy,
                    uav_load=dst_uav.uav_load,
                    uav_speed=dst_uav.uav_speed
                )

                src_uav.appendUAVNeighbor(uav_neighbor)
                self.uav_net[src_uav_id].add(dst_uav_id)

    def updateEachUavNeighbors(self, min_uav2uav_trans_rate: float = global_min_uav2uav_trans_rate):
        """
        不考虑 Base，更新每个 UAV 的 uav_neighbor_set，
        同时构建 Monitor 的 UAV 网络图 self.uav_net。

        判定 UAV j 是否可以成为 UAV i 的邻居：
        1. i 和 j 不是同一架 UAV；
        2. 二者距离 < UAV i 的 communication_radius；
        3. 由 UAV i 调用 get_uav2uavTransRate(j) 得到的通信速率 > min_uav2uav_trans_rate。

        self.uav_net 是邻接表形式：
        {
            uav_i_id: {neighbor_1_id, neighbor_2_id, ...},
            ...
        }

        注意：
        该函数不会把 base.uavID=900 加入 self.uav_net。
        """

        # 每次更新前，先清空旧网络图
        # 这里只加入 self.uav_set 中的 UAV，不加入 Base，uav_net里只考虑active的UAV
        self.uav_net.clear()
        self.uav_net = {uavID: set() for uavID in self.uav_set.keys() if self.uav_set[uavID].is_active}

        # 清空每架 UAV 的旧邻居信息
        for uav in self.uav_set.values():
            uav.uav_neighbor_set.clear()
            uav.num_UAV_neighbor = 0

        # 遍历每一对 UAV
        for src_uav_id, src_uav in self.uav_set.items():
            if not self.uav_set[src_uav_id].is_active:
                continue
            # 清空 src_uav 到其他 UAV 的传输速率字典
            src_uav.uav2uav_TransRate.clear()

            for dst_uav_id, dst_uav in self.uav_set.items():

                # UAV 不能把自己作为邻居
                if src_uav_id == dst_uav_id:
                    continue
                # dst必须是active的
                if not self.uav_set[dst_uav_id].is_active:
                    continue

                dx = dst_uav.uav_position[0] - src_uav.uav_position[0]
                dy = dst_uav.uav_position[1] - src_uav.uav_position[1]
                dz = dst_uav.uav_position[2] - src_uav.uav_position[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 条件 1：距离必须小于当前 UAV 的通信半径
                if distance > src_uav.communication_radius:
                    continue

                # 条件 2：通信速率必须大于阈值
                trans_rate = src_uav.getUav2UavTransRate(
                    dst_uav_id,
                    dst_uav.uav_position
                )

                if trans_rate < min_uav2uav_trans_rate:
                    continue

                # 构造邻居信息
                uav_neighbor = UavNeighbor(
                    uavID=dst_uav.uavID,
                    uav_type=dst_uav.uav_type,
                    uav_position=dst_uav.uav_position,
                    uav_energy=dst_uav.uav_energy,
                    uav_load=dst_uav.uav_load,
                    uav_speed=dst_uav.uav_speed
                )

                # 更新 src_uav 的邻居集合
                src_uav.appendUAVNeighbor(uav_neighbor)

                # 更新 Monitor 中的邻接表
                self.uav_net[src_uav_id].add(dst_uav_id)

    def updateAllAboutUAVnet(self,
                             min_energy=global_min_uav_energy,
                             has_bas=False,
                             min_uav2uav_trans_rate=global_min_uav2uav_trans_rate):
        self.updateAllUavActivities(min_energy)
        if has_bas:
            self.updateEachUavNeighborsWithBase(min_uav2uav_trans_rate)
        else:
            self.updateEachUavNeighbors(min_uav2uav_trans_rate)

    def checkUavnetConnectivity(self):
        """
                检查 UAV 网络是否全连通，并找出所有连通子图。

                返回:
                    is_connected: bool
                        True 表示整个 UAV 网络全连通，False 表示不全连通。

                    connected_components: list[set]
                        每个 set 是一个连通子图中的 UAV ID 集合。
                        例如:
                        [
                            {900, 1, 2, 3},
                            {4, 5},
                            {6}
                        ]
        """

        if not self.uav_net:
            return True, []

        # 如果希望按“无向图”检查连通性，先构造无向邻接表
        undirected_net = {uav_id: set(neighbors) for uav_id, neighbors in self.uav_net.items()}

        for uav_id, neighbors in self.uav_net.items():  # 有UAV1->UAV2，但是没有UAV2->UAV1，则添加UAV2->UAV1
            for neighbor_id in neighbors:
                if neighbor_id not in undirected_net:
                    undirected_net[neighbor_id] = set()
                undirected_net[neighbor_id].add(uav_id)  # python的集合里，相同添加已有元素是不会重复添加的，{1, 2, 2}就是{1, 2}

        visited = set()
        connected_components = []

        for start_uav_id in undirected_net.keys():
            if start_uav_id in visited:
                continue

            # BFS 搜索一个连通子图
            component = set()
            queue = [start_uav_id]
            visited.add(start_uav_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                for neighbor_id in undirected_net[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            connected_components.append(component)

        is_connected = len(connected_components) == 1

        if is_connected:
            print(f"\033[92m[UAV NET] 网络全连通，共 {len(undirected_net)} 个节点。\033[0m")
        # else:
        #     print(f"\033[91m[UAV NET] 网络不全连通，共有 {len(connected_components)} 个连通子图。\033[0m")
        #     for idx, component in enumerate(connected_components):
        #         print(f"  连通子图 {idx + 1}: {sorted(component)}")

        return is_connected, connected_components

    def getActiveMask(self):
        """
        获得active_mask，依据是能量、负载等因素决定是否active
        :return:形如[1 or 0, 1, 1, 1, 1, ...]的active_mask
        """
        active_mask = []
        for uav in self.uav_set.values():
            active_mask.append(uav.isActive())
        return active_mask

    def getState(self,
                 max_num_uav=None,
                 x_range=(0.0, 10000.0),
                 y_range=(0.0, 10000.0),
                 z_range=(0.0, 1000.0),
                 max_speed=1000.0,
                 max_comm_radius=2000.0,
                 include_distance_matrix=True,
                 auto_update_neighbors=False,
                 min_uav2uav_trans_rate=global_min_uav2uav_trans_rate,
                 return_info=False):
        """
        返回当前 UAV 网络状态，作为强化学习智能体的神经网络输入。

        返回:
            state: list[float]
                一维向量，可直接转换为 torch.tensor(state, dtype=torch.float32)

        state 结构:
            [global_features,
             node_features,
             adjacency_matrix_flatten,
             distance_matrix_flatten 可选]

        注意:
            1. 智能体不需要访问 Monitor。
            2. 智能体只接收本函数返回的 state。
            3. max_num_uav 应在训练过程中保持固定，否则神经网络输入维度会变化。
        """

        # 是否在取状态前自动更新邻接关系
        # 如果你在 monitor 执行动作后已经手动调用 updateEachUavNeighbors()，
        # 这里可以保持 False。
        if auto_update_neighbors:
            self.updateEachUavNeighbors(
                min_uav2uav_trans_rate=min_uav2uav_trans_rate
            )

        # ---------- 基本工具函数 ----------

        def safe_div(a, b):
            if b == 0:
                return 0.0
            return float(a) / float(b)

        def norm_value(v, value_range):
            low, high = value_range
            if high == low:
                return 0.0
            value = (float(v) - float(low)) / (float(high) - float(low))
            # 裁剪到 [0, 1]，避免异常坐标导致神经网络输入过大
            return max(0.0, min(1.0, value))

        def get_uav_position(uav):
            x, y, z = uav.uav_position
            return float(x), float(y), float(z)

        def is_uav_active(uav_id):
            """
            判断 UAV 是否仍然有效。
            damaged_uav_ids 中的 UAV 不参与连通性状态。
            """
            if uav_id not in self.uav_set:
                return False

            if uav_id in self.damaged_uav_ids:
                return False

            if self.remaining_uav_ids and uav_id not in self.remaining_uav_ids:
                return False

            uav = self.uav_set[uav_id]

            if hasattr(uav, "getIsActive"):
                return bool(uav.isActive())

            return bool(getattr(uav, "is_active", True))

        # ---------- UAV 顺序与最大数量 ----------

        all_uav_ids = sorted(self.uav_set.keys())

        if max_num_uav is None:
            max_num_uav = len(all_uav_ids)

        if len(all_uav_ids) > max_num_uav:
            raise ValueError(
                f"当前 UAV 数量为 {len(all_uav_ids)}，超过 max_num_uav={max_num_uav}。"
                f"请增大 max_num_uav，或者保持训练场景 UAV 数量固定。"
            )

        # slot_uav_ids 决定 state 中每个 UAV 的槽位顺序
        # 后续 monitor 执行动作时也应使用相同的 sorted(self.uav_set.keys()) 顺序
        slot_uav_ids = all_uav_ids + [None] * (max_num_uav - len(all_uav_ids))
        id_to_slot = {uav_id: idx for idx, uav_id in enumerate(all_uav_ids)}

        active_uav_ids = [
            uav_id for uav_id in all_uav_ids
            if is_uav_active(uav_id)
        ]

        active_uav_id_set = set(active_uav_ids)

        # ---------- 构造无向邻接矩阵 ----------

        adjacency_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        for src_id in all_uav_ids:
            if src_id not in active_uav_id_set:
                continue

            src_slot = id_to_slot[src_id]

            for dst_id in self.uav_net.get(src_id, set()):
                if dst_id not in id_to_slot:
                    continue
                if dst_id not in active_uav_id_set:
                    continue
                if dst_id == src_id:
                    continue

                dst_slot = id_to_slot[dst_id]

                # 连通性判断通常按无向图处理，所以这里也转成无向邻接矩阵
                adjacency_matrix[src_slot][dst_slot] = 1.0
                adjacency_matrix[dst_slot][src_slot] = 1.0

        # ---------- 静默计算连通子图，不调用 checkUavnetConnectivity，避免训练时频繁 print ----------

        visited = set()
        connected_components = []

        for start_id in active_uav_ids:
            if start_id in visited:
                continue

            component = set()
            queue = [start_id]
            visited.add(start_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                current_slot = id_to_slot[current_id]

                for other_id in active_uav_ids:
                    if other_id in visited:
                        continue

                    other_slot = id_to_slot[other_id]

                    if adjacency_matrix[current_slot][other_slot] > 0.5:
                        visited.add(other_id)
                        queue.append(other_id)

            connected_components.append(component)

        component_size_by_id = {}
        for component in connected_components:
            component_size = len(component)
            for uav_id in component:
                component_size_by_id[uav_id] = component_size

        num_active_uav = len(active_uav_ids)
        num_components = len(connected_components)
        largest_component_size = 0

        if connected_components:
            largest_component_size = max(len(c) for c in connected_components)

        is_connected = 1.0 if num_active_uav > 0 and num_components == 1 else 0.0

        # ---------- 全局网络特征 ----------

        edge_count = 0
        for i in range(max_num_uav):
            for j in range(i + 1, max_num_uav):
                if adjacency_matrix[i][j] > 0.5:
                    edge_count += 1

        max_possible_edges = num_active_uav * (num_active_uav - 1) / 2
        edge_density = safe_div(edge_count, max_possible_edges)

        avg_degree = safe_div(2 * edge_count, num_active_uav)

        global_features = [
            safe_div(num_active_uav, max_num_uav),
            safe_div(len(self.damaged_uav_ids), max_num_uav),
            safe_div(num_components, max_num_uav),
            safe_div(largest_component_size, max_num_uav),
            is_connected,
            edge_density,
            safe_div(avg_degree, max(max_num_uav - 1, 1)),
        ]

        # ---------- 节点特征 ----------

        node_features = []

        for slot_idx, uav_id in enumerate(slot_uav_ids):
            if uav_id is None:
                # padding UAV
                node_features.extend([
                    0.0,  # exists
                    0.0,  # active
                    0.0,  # x
                    0.0,  # y
                    0.0,  # z
                    0.0,  # energy
                    0.0,  # load
                    0.0,  # speed
                    0.0,  # communication radius
                    0.0,  # degree
                    0.0,  # component size
                ])
                continue

            uav = self.uav_set[uav_id]
            x, y, z = get_uav_position(uav)

            exists = 1.0
            active = 1.0 if uav_id in active_uav_id_set else 0.0

            degree = sum(adjacency_matrix[slot_idx])
            component_size = component_size_by_id.get(uav_id, 0)

            energy_norm = safe_div(
                getattr(uav, "uav_energy", 0.0),
                max(getattr(uav, "MAX_ENERGY", 1.0), 1.0)
            )

            load_norm = safe_div(
                getattr(uav, "uav_load", 0.0),
                max(getattr(uav, "MAX_LOAD", 1.0), 1.0)
            )

            speed_norm = safe_div(
                getattr(uav, "uav_speed", 0.0),
                max(max_speed, 1.0)
            )

            radius_norm = safe_div(
                getattr(uav, "communication_radius", 0.0),
                max(max_comm_radius, 1.0)
            )

            degree_norm = safe_div(
                degree,
                max(max_num_uav - 1, 1)
            )

            component_size_norm = safe_div(
                component_size,
                max_num_uav
            )

            node_features.extend([
                exists,
                active,
                norm_value(x, x_range),
                norm_value(y, y_range),
                norm_value(z, z_range),
                energy_norm,
                load_norm,
                speed_norm,
                radius_norm,
                degree_norm,
                component_size_norm,
            ])

        # ---------- 邻接矩阵展平 ----------

        adjacency_features = []
        for i in range(max_num_uav):
            for j in range(max_num_uav):
                adjacency_features.append(adjacency_matrix[i][j])

        # ---------- 距离矩阵展平，可选 ----------

        distance_features = []

        if include_distance_matrix:
            x_len = x_range[1] - x_range[0]
            y_len = y_range[1] - y_range[0]
            z_len = z_range[1] - z_range[0]
            max_distance = math.sqrt(x_len ** 2 + y_len ** 2 + z_len ** 2)

            for i in range(max_num_uav):
                uav_i_id = slot_uav_ids[i]

                for j in range(max_num_uav):
                    uav_j_id = slot_uav_ids[j]

                    if uav_i_id is None or uav_j_id is None:
                        distance_features.append(0.0)
                        continue

                    if uav_i_id not in active_uav_id_set or uav_j_id not in active_uav_id_set:
                        distance_features.append(0.0)
                        continue

                    uav_i = self.uav_set[uav_i_id]
                    uav_j = self.uav_set[uav_j_id]

                    xi, yi, zi = get_uav_position(uav_i)
                    xj, yj, zj = get_uav_position(uav_j)

                    dx = xi - xj
                    dy = yi - yj
                    dz = zi - zj

                    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    distance_features.append(safe_div(distance, max_distance))

        # ---------- 合并成最终 state ----------

        state = (
                global_features
                + node_features
                + adjacency_features
                + distance_features
        )

        if return_info:
            info = {
                "uav_order": all_uav_ids,
                "slot_uav_ids": slot_uav_ids,
                "num_active_uav": num_active_uav,
                "num_components": num_components,
                "largest_component_size": largest_component_size,
                "is_connected": bool(is_connected),
                "edge_count": edge_count,
                "state_dim": len(state),
                "node_feature_dim": 11,
                "global_feature_dim": len(global_features),
                "include_distance_matrix": include_distance_matrix,
            }
            return state, info

        return state

    def getAllState(self,
                    max_num_uav=None,
                    x_range=(0.0, 10000.0),
                    y_range=(0.0, 10000.0),
                    z_range=(0.0, 1000.0),
                    max_speed=1000.0,
                    max_comm_radius=2000.0,
                    include_distance_matrix=True,
                    auto_update_neighbors=False,
                    min_uav2uav_trans_rate=global_min_uav2uav_trans_rate):
        """
        返回当前 UAV 网络的完整结构化状态。

        返回:
            {
                "flat_state": flat_state,
                "global_features": global_features,
                "node_features": node_features,
                "adjacency_matrix": adjacency_matrix,
                "distance_matrix": distance_matrix,
            }

        其中:
            global_features: list[float]
                全局网络特征，形状为 [7]

            node_features: list[list[float]]
                每架 UAV 的节点特征，形状为 [max_num_uav, 11]

            adjacency_matrix: list[list[float]]
                UAV 网络邻接矩阵，形状为 [max_num_uav, max_num_uav]

            distance_matrix: list[list[float]]
                UAV 间距离矩阵，形状为 [max_num_uav, max_num_uav]

            flat_state: list[float]
                展平后的一维状态，可直接输入 MLP 强化学习网络
        """

        # 是否自动更新邻接关系
        if auto_update_neighbors:
            self.updateEachUavNeighbors(
                min_uav2uav_trans_rate=min_uav2uav_trans_rate
            )

        # ---------- 基本工具函数 ----------

        def safe_div(a, b):
            if b == 0:
                return 0.0
            return float(a) / float(b)

        def norm_value(v, value_range):
            low, high = value_range
            if high == low:
                return 0.0

            value = (float(v) - float(low)) / (float(high) - float(low))

            # 裁剪到 [0, 1]
            return max(0.0, min(1.0, value))

        def get_uav_position(uav):
            x, y, z = uav.uav_position
            return float(x), float(y), float(z)

        def is_uav_active(uav_id):
            """
            判断 UAV 是否仍然有效。
            damaged_uav_ids 中的 UAV 不参与连通性状态。
            """
            if uav_id not in self.uav_set:
                return False

            if uav_id in self.damaged_uav_ids:
                return False

            if self.remaining_uav_ids and uav_id not in self.remaining_uav_ids:
                return False

            uav = self.uav_set[uav_id]

            if hasattr(uav, "getIsActive"):
                return bool(uav.isActive())

            return bool(getattr(uav, "is_active", True))

        # ---------- UAV 顺序与最大数量 ----------

        all_uav_ids = sorted(self.uav_set.keys())

        if max_num_uav is None:
            max_num_uav = len(all_uav_ids)

        if len(all_uav_ids) > max_num_uav:
            raise ValueError(
                f"当前 UAV 数量为 {len(all_uav_ids)}，超过 max_num_uav={max_num_uav}。"
                f"增大 max_num_uav，或者保持训练场景中UAV数量固定。"
            )

        # 每个 UAV 的固定槽位
        slot_uav_ids = all_uav_ids + [None] * (max_num_uav - len(all_uav_ids))
        id_to_slot = {uav_id: idx for idx, uav_id in
                      enumerate(all_uav_ids)}  # UAV的ID，不一定连续、不一定由小到大，所以这里做了一个UAV的ID到索引的映射。

        active_uav_ids = [
            uav_id for uav_id in all_uav_ids
            if is_uav_active(uav_id)
        ]

        active_uav_id_set = set(active_uav_ids)

        # ---------- 构造无向邻接矩阵 ----------

        adjacency_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        for src_id in all_uav_ids:
            if src_id not in active_uav_id_set:  # 非active的UAV，不计入。
                continue

            src_slot = id_to_slot[src_id]

            for dst_id in self.uav_net.get(src_id, set()):
                if dst_id not in id_to_slot:
                    continue
                if dst_id not in active_uav_id_set:
                    continue
                if dst_id == src_id:
                    continue

                dst_slot = id_to_slot[dst_id]

                # 按无向图处理
                adjacency_matrix[src_slot][dst_slot] = 1.0
                adjacency_matrix[dst_slot][src_slot] = 1.0

        # ---------- 静默计算连通子图 ----------
        # 只统计active节点间的连通子图
        visited = set()
        connected_components = []

        for start_id in active_uav_ids:
            if start_id in visited:
                continue

            component = set()
            queue = [start_id]
            visited.add(start_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                current_slot = id_to_slot[current_id]

                for other_id in active_uav_ids:
                    if other_id in visited:
                        continue

                    other_slot = id_to_slot[other_id]

                    if adjacency_matrix[current_slot][other_slot] > 0.5:
                        visited.add(other_id)
                        queue.append(other_id)

            connected_components.append(component)

        component_size_by_id = {}

        for component in connected_components:
            component_size = len(component)
            for uav_id in component:
                component_size_by_id[uav_id] = component_size

        num_active_uav = len(active_uav_ids)
        num_components = len(connected_components)

        if connected_components:
            largest_component_size = max(len(c) for c in connected_components)
        else:
            largest_component_size = 0

        is_connected = 1.0 if num_active_uav > 0 and num_components == 1 else 0.0

        # ---------- 全局网络特征 ----------

        edge_count = 0

        for i in range(max_num_uav):
            for j in range(i + 1, max_num_uav):
                if adjacency_matrix[i][j] > 0.5:
                    edge_count += 1

        max_possible_edges = num_active_uav * (num_active_uav - 1) / 2
        edge_density = safe_div(edge_count, max_possible_edges)
        avg_degree = safe_div(2 * edge_count, num_active_uav)

        global_features = [
            safe_div(num_active_uav, max_num_uav),
            safe_div(len(self.damaged_uav_ids), max_num_uav),
            safe_div(num_components, max_num_uav),
            safe_div(largest_component_size, max_num_uav),
            is_connected,
            edge_density,
            safe_div(avg_degree, max(max_num_uav - 1, 1)),
        ]

        # ---------- 节点特征矩阵 [max_num_uav, 11] ----------

        node_features = []

        for slot_idx, uav_id in enumerate(slot_uav_ids):
            if uav_id is None:
                # padding UAV
                node_features.append([
                    0.0,  # exists
                    0.0,  # active
                    0.0,  # x
                    0.0,  # y
                    0.0,  # z
                    0.0,  # energy
                    0.0,  # load
                    0.0,  # speed
                    0.0,  # communication radius
                    0.0,  # degree
                    0.0,  # component size
                ])
                continue

            uav = self.uav_set[uav_id]
            x, y, z = get_uav_position(uav)

            exists = 1.0
            active = 1.0 if uav_id in active_uav_id_set else 0.0

            degree = sum(adjacency_matrix[slot_idx])
            component_size = component_size_by_id.get(uav_id, 0)

            energy_norm = safe_div(
                getattr(uav, "uav_energy", 0.0),
                max(getattr(uav, "MAX_ENERGY", 1.0), 1.0)
            )

            load_norm = safe_div(
                getattr(uav, "uav_load", 0.0),
                max(getattr(uav, "MAX_LOAD", 1.0), 1.0)
            )

            speed_norm = safe_div(
                getattr(uav, "uav_speed", 0.0),
                max(max_speed, 1.0)
            )

            radius_norm = safe_div(
                getattr(uav, "communication_radius", 0.0),
                max(max_comm_radius, 1.0)
            )

            degree_norm = safe_div(
                degree,
                max(max_num_uav - 1, 1)
            )

            component_size_norm = safe_div(
                component_size,
                max_num_uav
            )

            node_features.append([
                exists,
                active,
                norm_value(x, x_range),
                norm_value(y, y_range),
                norm_value(z, z_range),
                energy_norm,
                load_norm,
                speed_norm,
                radius_norm,
                degree_norm,
                component_size_norm,
            ])

        # ---------- 距离矩阵 [max_num_uav, max_num_uav] ----------

        distance_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        if include_distance_matrix:
            x_len = x_range[1] - x_range[0]
            y_len = y_range[1] - y_range[0]
            z_len = z_range[1] - z_range[0]
            max_distance = math.sqrt(x_len ** 2 + y_len ** 2 + z_len ** 2)

            for i in range(max_num_uav):
                uav_i_id = slot_uav_ids[i]

                for j in range(max_num_uav):
                    uav_j_id = slot_uav_ids[j]

                    if uav_i_id is None or uav_j_id is None:
                        distance_matrix[i][j] = 0.0
                        continue

                    if uav_i_id not in active_uav_id_set or uav_j_id not in active_uav_id_set:
                        distance_matrix[i][j] = 0.0
                        continue

                    uav_i = self.uav_set[uav_i_id]
                    uav_j = self.uav_set[uav_j_id]

                    xi, yi, zi = get_uav_position(uav_i)
                    xj, yj, zj = get_uav_position(uav_j)

                    dx = xi - xj
                    dy = yi - yj
                    dz = zi - zj

                    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    distance_matrix[i][j] = safe_div(distance, max_distance)

        # ---------- 构造 flat_state ----------

        flat_node_features = []
        for row in node_features:
            flat_node_features.extend(row)

        flat_adjacency_matrix = []
        for row in adjacency_matrix:
            flat_adjacency_matrix.extend(row)

        flat_distance_matrix = []
        for row in distance_matrix:
            flat_distance_matrix.extend(row)

        flat_state = (
                global_features
                + flat_node_features
                + flat_adjacency_matrix
                + flat_distance_matrix
        )

        return {
            "flat_state": flat_state,
            "global_features": global_features,
            "node_features": node_features,
            "adjacency_matrix": adjacency_matrix,  # 只考虑active的UAV间的邻接矩阵
            "distance_matrix": distance_matrix,
        }

    def countReward(self, uav_slot, delta):
        """
        计算奖励函数，奖励函数的目标是让整个网络连通，在实现联通的基础上，要让移动距离和次数最小
        先做一个强制移动训练，必须移动UAV8至指定地点
        :return:奖励reward，actor给出的位移方向越接近目标位移方向，得到的奖励越大。必须移动UAV-8，否则直接返回惩罚
        """
        bridge_destination = (2100.0, 300.0, 100)
        if uav_slot != 7:
            return -100
        moving_uavID = uav_slot
        moving_UAV = self.uav_set[moving_uavID]
        moving_uav_position = self.uav_set[moving_uavID].uav_position
        # print(f"position before moving {moving_uav_position}")
        sac_direction = (float(delta[0, 0]), float(delta[0, 1]), float(delta[0, 2]))
        bridge_direction = (bridge_destination[0] - moving_uav_position[0],
                            bridge_destination[1] - moving_uav_position[1],
                            bridge_destination[2] - moving_uav_position[2])
        cos_theta = ((sac_direction[0] * bridge_direction[0] + sac_direction[1] * bridge_direction[1] + sac_direction[2] * bridge_direction[2]) /
                     (math.sqrt(sac_direction[0] ** 2 + sac_direction[1] ** 2 + sac_direction[2] ** 2) *
                      math.sqrt(bridge_direction[0] ** 2 + bridge_direction[1] ** 2 + bridge_direction[2] ** 2)))
        reward = 10 * cos_theta
        return reward

    def step(self, uav_slot, delta):
        """
        在actor给出动作后， 用于环境执行动作，
        :param uav_slot:UAV的槽号，从0开始
        :param delta:UAV的位移三元组
        :return:奖励reward，全连接时返回done=True
        """
        reward = self.countReward(uav_slot, delta)
        moving_uavID = int(uav_slot)
        moving_UAV = self.uav_set[moving_uavID]
        moving_uav_position = self.uav_set[moving_uavID].uav_position
        # print(f"position before moving {moving_uav_position}")
        destination = (moving_uav_position[0] + float(delta[0, 0]),
                       moving_uav_position[1] + float(delta[0, 1]),
                       moving_uav_position[2] + float(delta[0, 2]))
        self.moveUAV(uavID=moving_uavID, destination=destination, moving_duration=160.0)
        is_connected, connected_components = self.checkUavnetConnectivity()
        done = is_connected
        return reward, done

    def draw_env_2D(self, positions, iteration=1, time_step=1):
        fig, ax = plt.subplots(figsize=(8, 8))

        x_list = []
        y_list = []

        for key in positions.keys():
            if key != 7:
                x_list.append(positions[key][0])
                y_list.append(positions[key][1])

        ax.scatter(
            x_list,
            y_list,
            color="dodgerblue",
            s=150,
            marker="o",
            edgecolors="black",
        )

        x_list.clear()
        y_list.clear()

        for key in positions.keys():
            if key == 7:
                x_list.append(positions[key][0])
                y_list.append(positions[key][1])

        ax.scatter(
            x_list,
            y_list,
            color="red",
            s=150,
            marker="o",
            edgecolors="black",
        )

        # for index, (x, y, z) in enumerate(positions.values()):
        #     ax.text(
        #         x,
        #         y + 5.0,
        #         f"UAV {index}\nz={z:.1f} m",
        #         ha="center",
        #         fontsize=9,
        #     )

        # ax.set_xlabel("X coordinate (m)")
        # ax.set_ylabel("Y coordinate (m)")
        ax.set_title(f"UAV PositionsTop View: ITE {iteration:03d} time step {time_step:02d}")

        # 设置坐标轴范围
        ax.set_xlim(-500, 4500)
        ax.set_ylim(-500, 4500)

        # 设置主刻度间隔
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_major_locator(MultipleLocator(500))

        # 设置次刻度间隔
        # ax.xaxis.set_minor_locator(MultipleLocator(5))
        # ax.yaxis.set_minor_locator(MultipleLocator(5))

        # 主网格和次网格
        ax.grid(True, which="major", linewidth=0.8)
        # ax.grid(True, which="minor", linewidth=0.4, linestyle="--")

        # 使 x、y 方向单位长度相同
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(f"./R_images/env_2D_ITE{iteration:04d}_TIM{time_step:03d}.jpeg", dpi=100)
        # plt.show()



