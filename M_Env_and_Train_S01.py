from torch import optim
from E_EnvController import EnvController
import torch
import torch.nn.functional as F
from SAC_Architectures.SAC_ReplayBuffer import SACReplayBuffer
from SAC_Architectures.SAC_Architecture01 import HybridSACActor, HybridSACCritic

SAC_replay_buffer = SACReplayBuffer(capacity=1000)
MAX_NUM_UAV = 8
env_controller = EnvController()
probe_monitor = env_controller.build_monitor_scene()
probe_state = probe_monitor.getState(max_num_uav=MAX_NUM_UAV)
state_dim = len(probe_state)
actor = HybridSACActor(num_uav=MAX_NUM_UAV, state_dim=state_dim)
critic_1 = HybridSACCritic(num_uav=MAX_NUM_UAV, state_dim=state_dim)
critic_2 = HybridSACCritic(num_uav=MAX_NUM_UAV, state_dim=state_dim)
target_critic_1 = HybridSACCritic(num_uav=MAX_NUM_UAV, state_dim=state_dim)
target_critic_2 = HybridSACCritic(num_uav=MAX_NUM_UAV, state_dim=state_dim)


optimizer_actor = optim.Adam(actor.parameters(), lr=0.0001/2)
optimizer_critic_1 = optim.Adam(critic_1.parameters(), lr=0.0001/2)
optimizer_critic_2 = optim.Adam(critic_2.parameters(), lr=0.0001/2)

# =============== SAC超参数===============
NUM_ITERATIONS = 1000  # 训练轮次
TOTAL_TIME_STEPS = 200  # 每轮时间步数
actor_process_times_counter = 0
BATCH_SIZE = 50  # 一批训练样本数
EPOCHS_EVERY_TRAINING = 4  # 每次训练轮数
GAMMA = 0.99
ALPHA = 0.1  # 熵温度系数：越大越鼓励探索（离散 SAC 的 soft value: Q - alpha*logπ）
TAU = 0.01  # target 网络 soft update 系数
MAX_GRAD_NORM = 10.0
UPDATE_EVERY = 10  # 每多少个 decision step 更新一次（1 表示每次决策都更一次）  400
device = torch.device('cpu')
decision_step_counter = 0  # 一轮iteration中，时间步总数total_time_step是固定的，但是一个时间步内的决策步个数不固定，所以需要自行统计一轮iteration中的决策步个数
iterations_rewards = [0.0] * NUM_ITERATIONS
iterations_average_rewards = [0.0] * NUM_ITERATIONS
iteration_CumuTras = [0.0] * NUM_ITERATIONS
best_average_reward = -100

# ==============开始训练===================
for i in range(NUM_ITERATIONS):
    monitor = env_controller.build_monitor_scene()

    time_step_counter = 0
    now_state = monitor.getState()
    now_state = torch.tensor(now_state, dtype=torch.float32).unsqueeze(0)  # 调整成 (batch_size, input_state_dim)的形式
    # 采样阶段，最大时间步就是total_time_step
    for time_step in range(TOTAL_TIME_STEPS):
        time_step_counter += 1
        active_mask = monitor.getActiveMask()
        if not any(active_mask):
            print("No active UAV is available; ending this episode.")
            break

        with torch.no_grad():
            action, uav_slot, delta, log_prob = actor.sample(state=now_state, active_mask=active_mask)  # a
        actor_process_times_counter += 1
        reward, env_done = monitor.step(int(uav_slot.item()), delta.detach())
        if time_step % 5 == 0:
            uav_positions = {}
            for key in monitor.uav_set.keys():
                uav_positions[key] = monitor.uav_set[key].uav_position
            monitor.draw_env_2D(positions=uav_positions, iteration=i, time_step=time_step)

        next_state = monitor.getState()
        next_active_mask = monitor.getActiveMask()
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        now_state = next_state  # 当前的 next_state 就是下一个时间步的 now_state
        reached_time_limit = time_step == TOTAL_TIME_STEPS - 1
        done = bool(env_done or reached_time_limit)
        SAC_replay_buffer.push(state=now_state,
                               active_mask=active_mask,
                               uav_slot=uav_slot,
                               delta=delta,
                               reward=reward,
                               next_state=next_state,
                               next_active_mask=next_active_mask,
                               done=done, )
        iterations_rewards[i] += reward
        if done:
            iterations_average_rewards[i] = iterations_rewards[i] / time_step_counter
            break
        # print(f"iteration: {i} / {NUM_ITERATIONS}, time_step: {time_step} / {TOTAL_TIME_STEPS}")

        # SAC训练部分
        if actor_process_times_counter % UPDATE_EVERY == 0 and SAC_replay_buffer.size >= BATCH_SIZE:

            # actions, uav_slots, deltas, log_probs = actor.sample(state=batch_states, active_mask=batch_active_masks)
            # 进行epochs_every_training 轮数的训练
            for epoch in range(EPOCHS_EVERY_TRAINING):
                # 每一个训练轮次都从replay buffer里重新随机选取batch_size个样本
                batch_samples = SAC_replay_buffer.sample(batch_size=BATCH_SIZE)
                states = batch_samples["states"]
                active_masks = batch_samples["active_masks"]
                uav_slots = batch_samples["uav_slots"]
                deltas = batch_samples["deltas"]
                rewards = batch_samples["rewards"]
                next_states = batch_samples["next_states"]
                next_active_masks = batch_samples["next_active_masks"]
                dones = batch_samples["dones"]

                # ----------------------------- Critic target -----------------------------
                with torch.no_grad():
                    next_uav_probs, next_log_probs, next_deltas_all = actor.sample_all(
                        state=next_states,
                        # Existing Actor.forward() internally calls torch.tensor(mask).
                        # Passing a Python list avoids copying a Tensor with a warning.
                        active_mask=next_active_masks.tolist(),
                    )

                    target_q1_all = target_critic_1.q_all_uavs(
                        state=next_states,
                        delta_all=next_deltas_all,
                    )
                    target_q2_all = target_critic_2.q_all_uavs(
                        state=next_states,
                        delta_all=next_deltas_all,
                    )
                    target_min_q_all = torch.minimum(target_q1_all, target_q2_all)

                    # V(s') = sum_d pi(d|s') [Q(s',d,c_d) - alpha*log pi(d,c_d|s')]
                    next_soft_value = (
                            next_uav_probs * (target_min_q_all - ALPHA * next_log_probs)
                    ).sum(dim=-1)

                    q_target = rewards + GAMMA * (1.0 - dones) * next_soft_value
                # ------------------------------ Twin Critics -----------------------------
                current_q1 = critic_1(states, uav_slots, deltas)  # 调用的是forward()
                current_q2 = critic_2(states, uav_slots, deltas)

                critic_1_loss = F.mse_loss(current_q1, q_target)
                critic_2_loss = F.mse_loss(current_q2, q_target)

                optimizer_critic_1.zero_grad(set_to_none=True)
                optimizer_critic_2.zero_grad(set_to_none=True)
                (critic_1_loss + critic_2_loss).backward()

                torch.nn.utils.clip_grad_norm_(critic_1.parameters(), MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(critic_2.parameters(), MAX_GRAD_NORM)

                optimizer_critic_1.step()
                optimizer_critic_2.step()

                # --------------------------------- Actor ---------------------------------
                # Critic parameters do not need gradients during the Actor update, but the
                # gradient through Q with respect to the sampled displacement is retained.
                critic_1.requires_grad_(False)
                critic_2.requires_grad_(False)


                uav_probs, total_log_probs, deltas_all = actor.sample_all(
                    state=states,
                    active_mask=active_masks.tolist(),
                )

                q1_policy_all = critic_1.q_all_uavs(states, deltas_all)
                q2_policy_all = critic_2.q_all_uavs(states, deltas_all)
                min_q_policy_all = torch.minimum(q1_policy_all, q2_policy_all)

                # J_pi = E_d[alpha*log pi(d,c_d|s) - min(Q1,Q2)]
                actor_loss = (
                        uav_probs * (ALPHA * total_log_probs - min_q_policy_all)
                ).sum(dim=-1).mean()

                optimizer_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                optimizer_actor.step()

                critic_1.requires_grad_(True)
                critic_2.requires_grad_(True)

                # --------------------------- Target soft update --------------------------
                if epoch + 1 == EPOCHS_EVERY_TRAINING:
                    with torch.no_grad():
                        for target_param_1, source_param_1 in zip(
                                target_critic_1.parameters(),
                                critic_1.parameters(),
                        ):
                            target_param_1.lerp_(source_param_1, TAU)

                        for target_param_2, source_param_2 in zip(
                                target_critic_2.parameters(),
                                critic_2.parameters(),
                        ):
                            target_param_2.lerp_(source_param_2, TAU)
    iterations_average_rewards[i] = iterations_rewards[i] / time_step_counter
    print(f"iteration: {i} / {NUM_ITERATIONS}, iteration average reward: {iterations_average_rewards[i]}")
    if iterations_average_rewards[i] > best_average_reward:
        best_average_reward = iterations_average_rewards[i]
        best_actor = actor.state_dict()
        best_critic_1 = critic_1.state_dict()
        best_critic_2 = critic_2.state_dict()
        best_target_critic_1 = target_critic_1.state_dict()
        best_target_critic_2 = target_critic_2.state_dict()
        torch.save(
            {
                "actor": best_actor,
                "critic_1": best_critic_1,
                "critic_2": best_critic_2,
                "target_critic_1": best_target_critic_1,
                "target_critic_2": best_target_critic_2,
                "hyperparameters":{
                    "GAMMA": GAMMA,
                    "ALPHA": ALPHA,
                    "MAX_GRAD_NORM": MAX_GRAD_NORM,
                    "TAU": TAU,
                    "TOTAL_TIME_STEPS": TOTAL_TIME_STEPS,
                    "NUM_ITERATIONS": NUM_ITERATIONS,

                }

            },
            f"./R_models/best_model_BAR_{best_average_reward+100:.4f}_ITE{i}.pth",
        )

    del monitor
    with open("./R_records/iterations_average_rewards.txt", "w") as f:
        for average_reward in iterations_average_rewards:
            f.write(str(average_reward) + '\n')


print("END")