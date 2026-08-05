import math
from typing import Tuple
from enum import Enum, auto

from dataclasses import dataclass

global_n0 = 3.98e-21
global_rho = 1
global_min_uav2uav_trans_rate = 60.0
global_min_uav_energy = 0.1
Energy_Threshold = 0


class UAVType(Enum):
    ClusterHead = auto()
    BackupClusterHead = auto()
    Linker = auto()
    Explorer = auto()
    Normal = auto()
    BASE = 900


@dataclass
class UavNeighbor:
    uavID: int
    uav_type: UAVType
    uav_position: Tuple[float, float, float]
    uav_energy: float
    uav_load: float
    uav_speed: float


class UAV:
    def __init__(self,
                 uavID=1,
                 uav_position=(10, 10, 999),
                 uav_speed=10,
                 uav_transpower=0,
                 uav_bandwidth=0,
                 MAX_LOAD=100,
                 MAX_ENERGY=100,
                 uav_type=UAVType.Normal, ):
        self.uavID = uavID
        self.uav_position = uav_position
        self.uav_speed = uav_speed
        self.uav_TransPower = uav_transpower  # 发射功率
        self.uav_Bandwidth = uav_bandwidth
        self.uav2uav_TransRate: [int, float] = {}
        self.MAX_LOAD = MAX_LOAD
        self.MAX_ENERGY = MAX_ENERGY
        self.uav_type = uav_type
        self.uav_energy = MAX_ENERGY
        self.uav_load = 0
        self.communication_radius = 0
        self.serving_GroundUser_set = []
        self.num_UAV_neighbor = 0
        self.uav_neighbor_set: [UavNeighbor] = {}
        self.num_serving_GroundUser = 0
        self.is_active = True

    def uavMove(self, moving_duration: float, destination: Tuple[float, float, float]):
        if not self.is_active:
            print(f"\033[91m [ERROR] UAV {self.uavID} is not active, can not move \033[0m")
        dx = destination[0] - self.uav_position[0]
        dy = destination[1] - self.uav_position[1]
        dz = destination[2] - self.uav_position[2]

        distance_to_destination = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        moving_distance = self.uav_speed * moving_duration

        # 已经在目标点
        if distance_to_destination == 0:
            return self.uav_position

        # 本次移动足够到达目标点，直接停在目标点
        if moving_distance >= distance_to_destination:
            self.uav_position = destination
        else:
            ratio = moving_distance / distance_to_destination
            self.uav_position = (
                self.uav_position[0] + dx * ratio,
                self.uav_position[1] + dy * ratio,
                self.uav_position[2] + dz * ratio
            )

        return self.uav_position

    def getUav2UavTransRate(self, uavID, uav_position, n0=3.98e-21, rho=1, min_uav2uav_trans_rate: float = 60.0):
        if self.uav_Bandwidth == 0 or self.uav_TransPower == 0:
            self.uav2uav_TransRate[uavID] = 0
            return self.uav2uav_TransRate[uavID]

        dx = uav_position[0] - self.uav_position[0]
        dy = uav_position[1] - self.uav_position[1]
        dz = uav_position[2] - self.uav_position[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        h = rho / distance ** 2  # 信道衰落系数
        SINR = (h * self.uav_TransPower) / (n0 * self.uav_Bandwidth)
        uav2uav_TransRate = self.uav_Bandwidth * math.log2(1 + SINR)
        if uav2uav_TransRate >= min_uav2uav_trans_rate:
            self.uav2uav_TransRate[uavID] = uav2uav_TransRate
        return uav2uav_TransRate

    def appendServingGroundUser(self, groundUserID):
        if groundUserID not in self.serving_GroundUser_set:
            self.serving_GroundUser_set.append(groundUserID)
            self.num_serving_GroundUser += 1
        else:
            print(f"\033[91m [ERROR] UAV{self.uavID} has been serving GroundUser{groundUserID}, but append again\033[0m")

    def appendUAVNeighbor(self, uav_neighbor: UavNeighbor):
        if uav_neighbor.uavID not in self.uav_neighbor_set:
            self.num_UAV_neighbor += 1
            self.uav_neighbor_set[uav_neighbor.uavID] = uav_neighbor
        else:
            del self.uav_neighbor_set[uav_neighbor.uavID]
            self.uav_neighbor_set[uav_neighbor.uavID] = uav_neighbor
            # 每隔固定时间间隔，monitor会重新更新整个网络，该过程会调用该函数
            # print(f"\033[95m [Warning] UAV{self.uavID} has had UavNeighbor{uav_neighbor.uavID}, but append again\033[0m")

    def isActive(self):
        """用于设置和返回UAV是否active"""
        if self.is_active and self.uav_energy >= Energy_Threshold:
            self.is_active = True
            return True
        self.is_active = False
        return False

    def countDistance(self, position=(100, 100, 100)):

        dx = position[0] - self.uav_position[0]
        dy = position[1] - self.uav_position[1]
        dz = position[2] - self.uav_position[2]

        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        return distance

    def infoState(self):
        print(f"\033[96m[UAV STATE] UAV {self.uavID} \nPosition{self.uav_position} \nEnergy{self.uav_energy}\033[0m")



