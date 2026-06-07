# RLT Franka 真机适配思路

这份文档整理当前把 RLinf 里的 ManiSkill RLT 迁移到 Franka 真机的方案。目标是先把边界和数据语义说清楚，避免把仿真、真机、OpenPI、RLT 论文算法和现有 RLinf realworld 流程混在一起。

## 1. 结论先行

RLT 真机适配不应该复用 ManiSkill 的 joint schema。ManiSkill RLT 继续保持 `panda-qpos` 路线：

```text
state: 9D panda qpos
action: 8D joint action
image/wrist_image: ManiSkill 采集出来的 OpenPI 输入图像
```

Franka 真机应该单独定义一个 `rlt_franka_ee` 或类似 schema，复用 RLinf 现有 Franka/OpenPI policy 和 realworld env 能力：

```text
state/proprio: Franka EE/gripper canonical state
action: Franka EE/gripper canonical action
image: realworld 主视角 RGB
wrist_image: realworld 腕部视角 RGB
prompt: yaml 或数据字段显式注入
```

算法上以 RLT 论文和 `openpi-RLT/rlt_online_rl` 为准。HG-DAgger 只能作为真机工程流程参考，不能把 DAgger 的 supervised training objective 搬进 RLT。

## 2. 非目标

这些事情当前不作为目标：

- 不把 `rl_phase` 重新加回 ManiSkill。真机 critical phase 可以由人决定，但不应该塞进 ManiSkill 自动逻辑。
- 不把 Franka 的 action/state 语义改造成 ManiSkill joint replay。Franka 和 ManiSkill 的 replayability 不一样。
- 不为 peg insertion 写死任务逻辑。真机接口应该通用，任务相关内容通过 prompt、reward/success/done provider、reset/terminal signal 配置化。
- 不在 `rlinf/rlt` 下面新开一套全局 adapter。按 RLinf 现有规则，schema/adapter 应该落在 env、OpenPI policy、dataconfig、model wrapper 这些已有扩展点里。

## 3. RLT 算法边界

RLT Stage2 的核心是 frozen VLA + frozen RL token + lightweight actor-critic：

```text
x = [z_rl, proprio]
a_tilde = frozen VLA reference action chunk
a = actor(x, a_tilde)
Q = critic(x, action_chunk)
```

actor loss 按论文是：

```text
L_actor = -Q(x, a) + beta * ||a - a_tilde||^2
```

RLT 不是 DAgger，也不应该只保存 expert samples。human intervention 在 RLT 里是 off-policy replay 的一部分。openpi-RLT 的关键语义是：

- replay 同时保存 `ref_chunk` 和 `action_chunk`
- `action_chunk` 是实际执行动作
- `source/source_chunk` 标记动作来源：`BASE / RL / HUMAN / MIXED`
- human/mixed chunk 上，actor 的 BC target 应切到 human executed action，而不是继续贴 VLA reference
- critic 学所有来源的 executed action，包括 base VLA、RL actor、human intervention、mixed chunk

当前 RLinf RLT 需要注意两点算法差异：

- 现有 `ResidualActor` 是 `a = a_tilde + residual`，严格说和论文/openpi-RLT 的 direct chunk actor 不完全一致。
- 现有 actor BC target 主要是 `a_tilde`，真机 intervention 接入后需要补 `source/source_chunk/intervention_flag` 语义，让 human/mixed 样本的 BC target 对齐实际执行动作。

## 4. 现有 RLinf Franka 采集缺口

RLinf Franka 文档里的数据采集主要服务于 SAC/RLPD CNN prior data。它已有：

- realworld env 的 state/image/action/reward/done
- SpaceMouse/GELLO 等 intervention wrapper
- `info["intervene_action"]` 和 `info["intervene_flag"]`
- `CollectEpisode` 可导出 pickle 或 LeRobot 风格数据
- reward/success/done 可通过任务 wrapper 或键盘/reward model 提供

但对 RLT/OpenPI 还不够，主要缺：

- OpenPI 输入需要的图像规格和多视角命名，例如 `image` / `wrist_image`，通常不应只保留 `128x128` CNN 图。
- prompt/task 的稳定来源。固定任务也要有 prompt，可以由 yaml 注入，但必须明确。
- action representation 定义。必须明确 Franka action 是 absolute EE target 还是 delta EE command。
- VLA reference chunk，也就是 `a_tilde/ref_chunk`。
- RL token 可重算信息，至少要保存 observation + prompt + model/checkpoint/version；在线 replay 可直接存 `z_rl`。
- RLT chunk replay 字段：`action_chunk/ref_chunk/rewards/next_ref_chunk/done`。
- `source/source_chunk`，用于区分 base VLA、RL actor、人类接管和混合 chunk。
- 失败轨迹。SFT 可以只用成功 demo，但 Stage2 replay 需要失败/终止样本给 critic 学 value。

## 5. Franka canonical schema

Franka 真机侧最先要定的是 canonical action/state，不是文件格式。

建议先定义：

```text
robot_type: franka_panda
control_schema: franka_ee
action_dim: 7
action_names: [x, y, z, roll, pitch, yaw, gripper]
```

但必须进一步确认：

```text
action_representation: abs | delta
rotation_representation: euler | rot6d | quat-derived
gripper_representation: open_close | width | normalized
control_frequency_hz: e.g. 10 / 20 / 50
chunk_len: RLT stage2 chunk length
```

真机同学最好同时记录 raw 和 canonical 两套内容：

```text
raw_robot_state:
  tcp_pose
  tcp_vel
  tcp_force
  tcp_torque
  gripper_position
  gripper_open
  joint_positions
  joint_velocities

raw_command:
  原始发给控制器的命令

canonical_state:
  RLT/OpenPI 输入使用的 proprio

canonical_action:
  RLT/OpenPI 训练和回放使用的 action
```

这样后续即使 canonical representation 需要调整，也可以从 raw trace 重建。

## 6. 离线数据要求

SFT 和 Stage1 主要使用 LeRobot/OpenPI 风格数据。每个 step 至少需要：

```text
image: uint8 HWC RGB
wrist_image: uint8 HWC RGB
state: float32 [proprio_dim]
actions: float32 [action_dim]
prompt/task: string
timestamp
frame_index
episode_index
done
is_success
intervene_flag
```

建议成功 demo 用于 SFT 和 Stage1。失败 demo 不一定进 SFT，但应保留给 Stage2/off-policy 分析。

对现有 RLinf OpenPI 接入，Franka 可优先复用：

```text
rlinf/models/embodiment/openpi/policies/franka_policy.py
rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py
```

但需要根据 RLT 增加一个明确的 Franka RLT dataconfig，处理：

- prompt 从 yaml 或数据字段注入
- image/wrist image key 映射
- state/action dim 对齐
- abs/delta action transform
- output action dim 截断到 Franka canonical action dim

## 7. 在线 RLT replay 要求

Stage2 在线 RLT replay 不应只保存普通 step trajectory。chunk transition 至少需要：

```text
z_rl: [embedding_dim]
proprio: [proprio_dim]
ref_chunk: [chunk_len, action_dim]
action_chunk: [chunk_len, action_dim]
rewards: [chunk_len]
done: bool
next_z_rl: [embedding_dim]
next_proprio: [proprio_dim]
next_ref_chunk: [chunk_len, action_dim]
source: uint8
source_chunk: [chunk_len]
intervention_flag: bool
success: int/bool
episode_id: int
step_id: int
```

其中：

```text
ref_chunk: frozen VLA 参考动作
action_chunk: 实际执行动作
source_chunk: 每个 step 的动作来源
intervention_flag: 该 chunk 是否包含人类接管
```

source 语义建议和 openpi-RLT 对齐：

```text
BASE = 0     # base VLA / fallback reference
RL = 1       # RLT actor
HUMAN = 2    # 完全人类接管
MIXED = 3    # chunk 内混合 policy/human
```

这套 replay 可以由在线流程直接生成，也可以从 raw episode trace 离线构建。

## 8. 需要真机同学提供的接口

最小接口不是一个 task-specific env，而是一组通用能力：

```text
reset()
get_observation()
step(action)
chunk_step(action_chunk)
send_action(action)
get_latest_human_action()
get_intervention_state()
get_success_failure_done_signal()
```

观测需要包含：

```text
frames:
  main camera
  wrist camera

state:
  raw robot state
  canonical proprio

task/prompt:
  可由配置注入，但运行时要能进入 OpenPI observation
```

step/chunk_step 的 info 需要包含：

```text
intervene_action
intervene_flag
source
source_chunk
manual_done
success
failure
raw_action
canonical_action
action_clipped
```

如果真机只先实现数据采集，不实现在线 actor service，也至少要记录 enough raw trace，使我们后续可以重建：

```text
observation_t
action_t
reward_t
done_t
success_t
human_intervention_t
prompt
camera frames
raw/canonical robot state
```

## 9. RLinf 适配分层

建议后续按以下层改，不要把所有逻辑塞进 RLT policy：

### Env 层

负责真实机器人交互、动作安全、人工接管、reward/done/success。

已有基础：

```text
rlinf/envs/realworld/realworld_env.py
rlinf/envs/realworld/franka/franka_env.py
rlinf/envs/realworld/common/wrappers/*
```

需要补齐：

- RLT 所需 `source/source_chunk`
- chunk 内 human/mixed 细粒度记录
- action canonicalization 和 safety clipping 的日志
- 可选 raw episode trace 保存

### OpenPI policy/dataconfig 层

负责把 Franka 数据转成 OpenPI 输入输出。

已有基础：

```text
rlinf/models/embodiment/openpi/policies/franka_policy.py
rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py
```

需要补齐：

- RLT Franka schema
- prompt 注入
- image/wrist image key 对齐
- action dim 和 representation adapter

### RLT Stage2 层

负责 frozen VLA/RL token 特征、actor-critic、replay、TD3 更新。

已有基础：

```text
rlinf/models/embodiment/rlt_stage2/
rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py
rlinf/workers/rollout/hf/huggingface_worker.py
```

需要补齐：

- direct actor 或明确保留 residual actor 的算法选择
- replay buffer 保存 `source/source_chunk/intervention_flag`
- actor loss 的 human/mixed BC target switch
- stratified sampling，可选增加 human intervention ratio / warmup demo ratio
- action representation adapter

## 10. SFT / Stage1 / Stage2 的兼容路线

### SFT

输入是 Franka LeRobot/OpenPI 数据。

需要保证：

```text
image/wrist_image shape 对齐 OpenPI
state dim = Franka proprio_dim
actions dim = Franka action_dim
prompt 来源明确
action representation 和部署一致
```

### Stage1

训练 RL token，仍然吃 OpenPI/VLA observation。

需要保证：

```text
同 SFT 的 observation schema
prompt 必须一致
state/action dim 不要混用 ManiSkill joint schema
```

### Stage2

在线或离线构建 RLT replay。

需要保证：

```text
ref_chunk 来自 frozen VLA
action_chunk 是实际执行动作
z_rl/proprio 和 next_z_rl/next_proprio 对齐
rewards/done/success 是 chunk-level TD target 可用的
source/source_chunk/intervention_flag 完整
```

## 11. 推荐落地顺序

建议分四步走。

1. 固化 Franka canonical schema

先和真机同学确定 action/state/prompt/image 的字段和维度。这个不定，后面所有训练都是假对齐。

2. 做离线 Franka OpenPI 数据适配

先不接在线 RLT，只保证 Franka 采集数据能跑 SFT/Stage1，并能通过 dataconfig/policy shape check。

3. 做 RLT replay 转换

从 raw episode trace 或在线 rollout 生成 RLT chunk transition，先离线检查：

```text
ref_chunk/action_chunk shape
z_rl/proprio shape
done/reward alignment
source/source_chunk/intervention_flag
```

4. 接在线 Stage2

最后才接真实机器人在线 actor-critic。上线前必须有：

```text
action clipping
workspace limit
human override
manual success/failure/done
fallback to VLA reference
replay journal persistence
```

## 12. 当前需要确认的问题

后续和真机同学对接时，优先确认这些：

- Franka action 是 absolute EE target 还是 delta EE command？
- OpenPI Franka policy 当前训练数据用的 action representation 是什么？
- state/proprio 最终用 7D、更多 EE state，还是 raw state 拼接？
- 主视角和腕部相机命名、分辨率、时间同步方式是什么？
- prompt 是数据里存，还是 yaml 注入？
- 失败 episode 是否保存？
- human intervention 是 per-step 记录，还是只记录 chunk/episode 标志？
- success/failure/done 是任务 wrapper 自动判断，还是人工按键？
- online rollout 是否只优化 critical phase？如果是，critical phase 由人手动标记还是另有 policy 触发？

这些问题确认后，再写 `rlt_franka_ee` 的 dataconfig/env config 会更稳。
