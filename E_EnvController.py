import torch

from E_Monitor import Monitor, base
from E_UAV import UAV, UAVType
from SAC_Architectures.SAC_Architecture01 import HybridSACActor

COMMUNICATION_RADIUS = 2000.0
UAV_TRANS_POWER = 1.0
UAV_BANDWIDTH = 1e6
MIN_TRANS_RATE = 60.0
ALTITUDE = 100.0
BASE_ID = base.uavID

class EnvController:
    def __init__(self):
        print(f"EnvController is working")

    def make_uav(self, uav_id, position, speed=20.0):
        """
        创建一架 UAV。
        注意：你的 UAV 默认 communication_radius=0、发射功率=0、带宽=0，
        所以这里必须手动设置，否则无人机之间不会建立邻居关系。
        """
        uav = UAV(
            uavID=uav_id,
            uav_position=position,
            uav_speed=speed,
            uav_transpower=UAV_TRANS_POWER,
            uav_bandwidth=UAV_BANDWIDTH,
            MAX_LOAD=100,
            MAX_ENERGY=10000,
            uav_type=UAVType.Normal,
        )
        uav.communication_radius = COMMUNICATION_RADIUS
        return uav

    def update_network_without_base(self, monitor):
        """
        使用 Monitor 原有的 updateEachUavNeighbors() 更新网络，
        然后移除 base=900，使本场景只考虑 8 架普通 UAV。
        """
        monitor.updateEachUavNeighbors(min_uav2uav_trans_rate=MIN_TRANS_RATE)

        # 从邻接表中删除 base 节点
        monitor.uav_net.pop(BASE_ID, None)

        # 从所有 UAV 的邻接表里删除 base
        for neighbors in monitor.uav_net.values():
            neighbors.discard(BASE_ID)

        # 如果某架 UAV 的邻居对象集合里包含 base，也一并删除
        for uav in monitor.uav_set.values():
            if BASE_ID in uav.uav_neighbor_set:
                del uav.uav_neighbor_set[BASE_ID]
                uav.num_UAV_neighbor = max(0, uav.num_UAV_neighbor - 1)

        return monitor.uav_net

    def print_network_state(self, title, monitor):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        print("UAV positions:")
        for uav_id in sorted(monitor.uav_set.keys()):
            print(f"  UAV{uav_id}: {monitor.uav_set[uav_id].uav_position}")

        print("\nUAV network adjacency list:")
        for uav_id in sorted(monitor.uav_net.keys()):
            print(f"  UAV{uav_id}: {sorted(monitor.uav_net[uav_id])}")

        is_connected, components = monitor.checkUavnetConnectivity()

        print(f"\nIs connected? {is_connected}")
        print(f"Connected components: {[sorted(c) for c in components]}")

        return is_connected, components

    def build_monitor_scene(self):
        """
        构造初始场景：

        第 1 组：4 架 UAV
            UAV1, UAV2, UAV3, UAV4

        第 2 组：3 架 UAV
            UAV5, UAV6, UAV7

        第 3 组：1 架 UAV
            UAV8

        初始状态下三组互不连通。
        UAV8 移动到 (2100, 300, 100) 后，可以同时连接第 1 组和第 2 组。
        """

        # 把 base 放到很远处；虽然之后还会从 uav_net 中删除它，但这样更保险
        base.uav_position = (100000.0, 100000.0, ALTITUDE)

        monitor = Monitor()

        positions = {
            # 第 1 组：4 架，内部距离都小于 2000 m
            0: (0.0, 0.0, ALTITUDE),
            1: (600.0, 0.0, ALTITUDE),
            2: (0.0, 600.0, ALTITUDE),
            3: (600.0, 600.0, ALTITUDE),

            # 第 2 组：3 架，内部距离都小于 2000 m
            # 与第 1 组最近距离约 3000 m，因此初始不连通
            4: (3600.0, 0.0, ALTITUDE),
            5: (4200.0, 0.0, ALTITUDE),
            6: (3600.0, 600.0, ALTITUDE),

            # 第 3 组：1 架，初始位置远离前两组
            7: (2100.0, 3500.0, ALTITUDE),
        }

        for uav_id, pos in positions.items():
            monitor.addUAV(self.make_uav(uav_id, pos))

        return monitor









print("END")
