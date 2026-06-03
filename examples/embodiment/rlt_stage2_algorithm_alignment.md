# RLT Stage2 论文算法与本地实现对齐说明

这份说明对齐的是 RLT 论文里的 Stage2：冻结 VLA 和 RL token，用一个轻量 actor-critic 在 VLA 参考动作附近做在线强化学习微调。

核心一句话：

```text
Stage2 = frozen VLA + frozen RL token + residual actor + twin-Q critic + replay buffer + optional intervention
```

## 1. 符号表

| 符号 | 通俗解释 | 本地实现位置 |
| --- | --- | --- |
| `s` | 当前环境观测。可以理解成“机器人现在看到和感觉到什么”，包括图像、关节状态等。 | rollout 传入的 `env_obs` |
| `l` / `ell` | 语言指令。比如“insert the peg”。论文里常写成 script `ell`。 | VLA observation 里的 prompt |
| `z` | VLA 内部 transformer embedding。可以理解成“大模型脑子里的高维中间特征”。 | `Stage2VLAWrapper.extract_embeddings()` |
| `z_rl` | RL token 压缩后的特征。可以理解成“给 RL 用的小号状态摘要”。 | `RLTokenModel.encode()` |
| `s_p` | proprioceptive state，机器人本体状态。比如关节角、夹爪状态、末端状态。 | `observation.state[:, :proprio_dim]` |
| `x` | actor 和 critic 真正吃的 RL 状态，`x = [z_rl, s_p]`。可以理解成“视觉语言摘要 + 机器人自身状态”。 | `rlt_stage2_policy.py::_prepare_features()` |
| `a_tilde` / `ã` | VLA 给出的参考动作 chunk。可以理解成“base VLA 觉得接下来应该怎么动”。 | `vla.get_rl_chunk_reference()` |
| `a` | 实际执行或训练使用的动作 chunk。可能来自学生 actor、base VLA、或者 expert/human intervention。 | `Trajectory.actions` / replay buffer 字段 `a` |
| `a_human` | 人接管时给出的动作。仿真里可以用强 expert 模型代替人。 | `huggingface_worker.py` 里 expert action |
| `pi_vla` | base VLA policy。可以理解成“原始 VLA 策略”。 | rollout 的 VLA backbone |
| `pi_theta` | Stage2 actor policy。`theta` 是 actor 参数。可以理解成“要训练的小策略头”。 | `ResidualActor` |
| `Q_psi` | critic。`psi` 是 critic 参数。可以理解成“评价某个动作 chunk 好不好的打分器”。 | `TwinQCritic` |
| `Q_psi_prime` | target critic。可以理解成“慢慢跟随 critic 的稳定版本”，TD3 用它算 target。 | `q1_target` / `q2_target` |
| `B` | replay buffer。可以理解成“经验池”，存过去采集到的 transition。 | `RLTStage2ReplayBuffer` |
| `r` | reward。可以理解成“这一步有没有好结果”。当前 ManiSkill 配置是 sparse/only-success。 | env reward / `rewards` |
| `done` | episode 是否结束。结束后 TD target 不再 bootstrap。 | replay buffer 字段 `dones` |
| `gamma` / `γ` | 折扣因子。越接近 1，越重视未来 reward。当前是 `0.99`。 | YAML `gamma` |
| `C` | RL action chunk 长度。可以理解成“actor 一次输出未来几步动作”。当前 joint 配置是 `10`。 | YAML `num_action_chunks` |
| `H` | VLA 原生动作 horizon。论文里 VLA 可能产生更长参考 chunk，本地取其中 `C` 步作为 RL chunk。 | `get_rl_chunk_reference(..., chunk_length)` |
| `beta` / `β` | BC regularizer 权重。可以理解成“强迫 actor 别离 VLA 参考动作太远的力度”。 | `bc_weight` / `bc_regularizer_beta` |
| `alpha` / `α` | 论文 Algorithm 1 里的可选 VLA fine-tuning 权重。当前本地 Stage2 主流程不更新 VLA。 | 本地 Stage2 不使用 |
| `N_warm` | warmup 环境步数。论文里指先用 base VLA 采数据填 buffer。 | 本地用 `min_buffer_size` / `actor_warmup_steps` 近似 |
| `G` | update-to-data ratio。可以理解成“每采一批数据，训练更新几次”。 | `update_epoch` 和 runner 调度共同决定 |
| `tau` / `τ` | target network 软更新系数。越小 target critic 变化越慢。当前是 `0.005`。 | YAML `tau` |
| `sigma` / `σ` | 噪声强度。actor rollout 或 TD3 target smoothing 会用。 | `actor_noise_sigma` / `target_noise_sigma` |
| `L_Q` | critic loss。让 critic 预测值接近 TD target。 | `critic_loss()` |
| `L_actor` | actor loss。让 actor 一边拿高 Q，一边别离 VLA reference 太远。 | `actor_loss()` |
| `intervention_flags` | 标记某条 transition 是否由 expert/human 接管。 | rollout `forward_inputs` 和 replay buffer stats |

## 2. 论文 Stage2 总流程

论文 Stage2 的完整流程可以翻成下面这几步：

```text
1. 冻结 VLA 和 RL token。
2. 先用 base VLA 执行动作，采 N_warm 步数据放进 replay buffer。
3. 在线 rollout 时：
   3.1 VLA 给参考动作 a_tilde。
   3.2 RL token 给状态表示 z_rl。
   3.3 actor 根据 x=[z_rl,s_p] 和 a_tilde 输出动作 a。
   3.4 如果人接管，就执行 a_human，并把 replay buffer 里的 reference 也替换成 a_human。
4. 从 replay buffer 采样训练 critic。
5. 每训练若干次 critic，再训练一次 actor。
6. actor 训练时用 Q 信号提升动作，同时用 BC regularizer 约束动作不要离 a_tilde 太远。
7. 对一部分 batch 做 reference action dropout，防止 actor 只复制 a_tilde。
```

## 3. RL Token 和 VLA Reference

### 论文算法

论文先训练 RL token，把 VLA 的内部 embedding `z` 压成 `z_rl`。Stage2 训练时冻结 VLA 和 RL token，只拿它们做特征提取：

```text
z_rl = RLToken(VLA_embedding)
x = [z_rl, s_p]
a_tilde = pi_vla(s, l)
```

通俗说法：VLA 负责“看懂当前场景”和“给一个初始动作建议”，RL token 把 VLA 的理解压成小向量，actor-critic 只在这个小向量上学习。

### 本地实现

本地在 `RLTStage2Policy` 里加载 VLA 和 RL token：

```text
rlinf/models/embodiment/rlt_stage2/rlt_stage2_policy.py
```

关键函数：

```text
_prepare_features()
```

它做了四件事：

```text
1. prepare_obs(env_obs)
2. extract_embeddings(observation)
3. rl_token_model.encode(embeddings, pad_mask)
4. get_rl_chunk_reference(observation, chunk_length)
```

### 为什么这么写

这样 actor worker 训练时不用反复跑大 VLA。rollout 负责缓存 `x` 和 `a_tilde`，actor worker 只训练小 actor 和小 critic，显存和速度都更可控。

### 对齐程度

这一块基本对齐。Stage2 使用的是 frozen VLA + frozen RL token。

## 4. Actor：在 VLA 参考动作附近做 residual 修正

### 论文算法

论文 actor 是：

```text
pi_theta(a_1:C | x, a_tilde_1:C)
```

意思是 actor 看当前状态 `x` 和 VLA 参考动作 `a_tilde`，输出接下来 `C` 步动作 `a`。

### 本地实现

本地 actor 是 residual 形式：

```text
a = a_tilde + residual_actor(x, a_tilde)
```

位置：

```text
rlinf/models/embodiment/rlt_stage2/components.py
```

类：

```text
ResidualActor
```

它的最后一层被 zero init，所以训练刚开始时：

```text
residual ≈ 0
a ≈ a_tilde
```

### 为什么这么写

Stage2 的目标不是从零学机器人控制，而是在 base VLA 已经会的动作附近微调。zero residual 让训练一开始就是 base VLA 行为，避免第一步就随机探索炸掉。

### 对齐程度

核心思想对齐。但论文写的是 Gaussian action distribution，本地实现更接近 deterministic TD3 actor，加少量 action noise。这是工程化 TD3 写法，不是论文公式的逐字实现。

## 5. Critic：chunk-level TD3

### 论文算法

critic 估计：

```text
Q_psi(x, a_1:C)
```

意思是给定当前状态和一整个动作 chunk，评价这个 chunk 未来能拿多少 return。

TD target 是：

```text
target = r_1 + gamma*r_2 + ... + gamma^(C-1)*r_C + gamma^C * Q_target(x_next, a_next)
```

通俗说法：先把这个 chunk 里已经发生的 reward 加起来，再用 target critic 估计 chunk 之后的未来价值。

### 本地实现

位置：

```text
rlinf/models/embodiment/rlt_stage2/components.py
```

关键函数：

```text
compute_td_target()
critic_loss()
TwinQCritic
```

本地也做了 TD3 的 target policy smoothing：

```text
next_a = actor(next_x, next_a_tilde)
next_a = next_a + clipped_noise
next_q = min(q1_target, q2_target)
```

### 为什么这么写

PegInsertion 这类任务 reward sparse，单步 credit assignment 很难。chunk-level target 让 critic 一次看一段动作，学习信号更短、更稳定。Twin-Q 和 target smoothing 是 TD3 防止 Q 过估计和训练抖动的常规设计。

### 对齐程度

基本对齐 TD3 chunk critic。

## 6. Actor Loss：Q 提升 + BC 约束

### 论文算法

论文 actor 一边要让 critic 觉得动作好，一边要保持靠近 VLA reference：

```text
maximize Q_psi(x, a)
regularize a close to a_tilde with beta
```

通俗说法：actor 可以偏离 VLA，但不能乱飞。

### 本地实现

位置：

```text
rlinf/models/embodiment/rlt_stage2/components.py
```

本地 loss：

```text
L_actor = bc_weight * MSE(a, a_tilde) - q_weight * Q(x, a) + delta_weight * delta_loss
```

各项白话解释：

| 项 | 通俗解释 |
| --- | --- |
| `MSE(a, a_tilde)` | actor 动作和 VLA 参考动作的距离 |
| `bc_weight` | 距离惩罚力度，对应论文的 `beta` |
| `- q_weight * Q(x, a)` | 希望 Q 越大越好，所以 loss 里是负号 |
| `delta_loss` | 额外动作平滑项，让 chunk 内动作变化别太怪 |

YAML 里当前是：

```yaml
warmup_bc_weight: 7.0
warmup_q_weight: 0.05
online_bc_weight: 3.0
online_q_weight: 0.1
delta_weight: 0.0
```

### 为什么这么写

critic 早期不准，所以 warmup 阶段用更大的 BC 和更小的 Q，防止 actor 被坏 Q 值带偏。后面逐渐降低 BC、提高 Q，让 RL 信号开始发挥作用。

### 对齐程度

BC regularizer 对齐。分 warmup/online 权重是本地额外的稳定化工程设计。

## 7. Reference Action Dropout

### 论文算法

论文说 actor 可能只复制 `a_tilde`，所以训练 actor 时随机挑一部分 batch，把传给 actor 的 reference chunk 置零：

```text
actor_input = [x, zero_like(a_tilde)]
```

注意：这是“传给 actor 的输入 a_tilde 置零”，不是把 BC target 置零，也不是 rollout 时随机发零动作。

### 本地实现

位置：

```text
rlinf/models/embodiment/rlt_stage2/components.py
```

函数：

```text
_apply_ref_dropout()
```

YAML：

```yaml
ref_action_dropout: 0.5
```

### 为什么这么写

如果 actor 总能看到 `a_tilde`，同时 loss 又让它靠近 `a_tilde`，它很容易只学复制。dropout 逼 actor 学一条不完全依赖 VLA reference 的动作生成路径。

### 对齐程度

这一块对齐论文。

## 8. Replay Buffer

### 论文算法

buffer `B` 里混合存三类数据：

| 数据来源 | 通俗解释 |
| --- | --- |
| VLA warmup data | base VLA 自己跑出来的数据 |
| online RL rollout | 当前 actor 跑出来的数据 |
| human intervention | 人接管纠偏的数据 |

每条 transition 至少要存：

```text
x, a, a_tilde, reward, x_next, a_tilde_next, done
```

### 本地实现

位置：

```text
rlinf/models/embodiment/rlt_stage2/replay_buffer.py
```

字段：

```text
x
a
a_tilde
rewards
next_x
next_a_tilde
dones
intervention
```

rollout 转 transition 的地方：

```text
rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py::_trajectory_to_transitions()
```

### 为什么这么写

TD3 是 off-policy，过去的数据可以反复用。只要 buffer 里存了“当时执行的动作 `a`”和“当时用于约束/条件的 reference `a_tilde`”，critic 和 actor 就能从混合数据里学习。

### 对齐程度

数据结构基本对齐。

## 9. Human Intervention

### 论文算法

论文说人可以在 rollout 中接管：

```text
if human intervenes:
    execute a_human
    a_tilde_in_buffer = a_human
else:
    execute actor action
```

关键点：接管时不只是执行 `a_human`，还要把 replay buffer 里的 reference 替换成 `a_human`。

### 本地实现

本地用 expert model 模拟 human intervention。

位置：

```text
rlinf/workers/rollout/hf/huggingface_worker.py
```

RLT intervention 时：

```text
1. 先跑学生，拿到学生的 x 和 base a_tilde。
2. 再跑 expert，拿到 expert action。
3. 执行动作改成 expert action。
4. forward_inputs["a_tilde"] 也替换成 expert action。
5. intervention_flags 标记为 True。
```

YAML：

```yaml
intervention:
  enable: True
  success_metric: "env/success_once"
  baseline_warmup_steps: 10
  relative_drop: 0.10
  probability: 0.9
```

### 为什么这么写

如果只执行 expert action，但 `a_tilde` 仍然是差的 base VLA action，actor loss 会被 BC 拉回差动作，训练信号冲突。把 `a_tilde` 替换成 expert action，等价于告诉 actor：“这次纠偏动作就是新的参考动作。”

### 对齐程度

“接管后替换 reference”这一点对齐论文。差异是论文是真人接管，本地是 expert 模型接管；论文的人可以判断时机，本地用 success drop 和 probability 自动触发。

## 10. Warmup

### 论文算法

论文 warmup 是：

```text
for N_warm environment steps:
    execute base VLA action a_tilde
    store transitions into replay buffer B
```

通俗说法：先别让还没学会的 actor 控制机器人，先让 base VLA 跑一批数据，让 critic 有东西学。

### 本地实现

本地 joint Stage2 配置现在主要是四层近似：

| 配置 | 通俗解释 |
| --- | --- |
| `train_every_episodes: 400` | 攒够约 400 个完成 episode 后触发一轮大训练 |
| `train_every_transitions: 20000` | 如果 episode 结束慢，攒够 20000 条 chunk transition 也触发一轮大训练 |
| `replay_buffer.min_buffer_size: 600` | 任一 actor rank 的 buffer 少于 600 条 transition 时不训练 |
| `actor_warmup_steps: 600` | buffer 不够时不更新 actor |
| `actor_warmup_updates: 200` | 前 200 个 actor/critic update 使用 warmup loss 权重 |

相关 YAML：

```yaml
update_epoch: 20000
train_every_episodes: 400
train_every_transitions: 20000
actor_warmup_steps: 600
actor_warmup_updates: 200
actor_loss_ramp_updates: 400
replay_buffer:
  min_buffer_size: 600
```

### 为什么这么写

这是为了在 RLinf 现有 runner 里少改 rollout 流程，同时避免“每次 90s rollout 只做 1 次 critic update”的低 update-to-data ratio：先让 buffer 有一批数据，再一次性做 20000 次 critic update 和约 10000 次 actor update。

### 对齐程度

这一块不是严格对齐。论文要求 warmup 阶段显式执行 base VLA 的 `a_tilde`；本地目前主要是“训练侧 warmup”和“loss 侧 warmup”，没有保证 rollout 阶段一定执行 `a_tilde`。

如果要更接近论文，应加：

```text
if global_step < rollout_warmup_steps:
    action = a_tilde
else:
    action = actor(x, a_tilde)
```

## 11. Action Chunk Subsampling

### 论文算法

论文说虽然 actor 一次输出 `C` 步动作，但中间每一步也能拿 observation，所以可以按 stride 存更多 transition：

```text
<x_0, a_0:C>
<x_2, a_2:C+2>
<x_4, a_4:C+4>
...
```

通俗说法：一个长动作 chunk 不只贡献一条训练数据，可以滑动窗口多切几条。

### 本地实现

本地没有明确写死论文里的 stride=2。当前是 actor worker 把 rollout 传来的每个 `t` 都转成 transition。

位置：

```text
rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py::_trajectory_to_transitions()
```

### 为什么这么写

RLinf 的 env/rollout 已经按 chunk 执行和收集 trajectory，本地实现直接消费现有 trajectory 粒度，避免额外重组动作窗口。

### 对齐程度

思想部分对齐，但不是论文 stride=2 的精确实现。

## 12. Update Ratio 和 Weight Sync

### 论文算法

论文说用 off-policy actor-critic，可以高 update-to-data ratio，并且实践中两个 critic update 对一个 actor update。

### 本地实现

YAML：

```yaml
update_epoch: 20000
critic_updates_per_actor: 2
critic_actor_ratio: ${actor.model.rlt_stage2.critic_updates_per_actor}
train_every_episodes: 400
train_every_transitions: 20000
```

actor worker 中：

```text
每次 update 都训练 critic。
每隔 critic_actor_ratio 次才训练 actor。
run_training() 只有在完成 episode 或 transition 数达到阈值后才真正执行 update_epoch 次更新。
```

rollout 权重同步只同步 actor：

```text
ROLLOUT_SYNC_PREFIXES = ("actor.",)
```

### 为什么这么写

rollout 不需要 critic，critic 只用于训练。只同步 actor 能减少通信和显存压力。

### 对齐程度

critic:actor 比例对齐。异步程度取决于 RLinf runner 和当前 `weight_sync_interval`。

## 13. 当前本地 Stage2 对齐总结

| 论文模块 | 本地状态 | 结论 |
| --- | --- | --- |
| Frozen VLA | 已实现 | 对齐 |
| Frozen RL token | 已实现 | 对齐 |
| `x=[z_rl,s_p]` | 已实现 | 对齐 |
| VLA reference `a_tilde` | 已实现 | 对齐 |
| residual actor | 已实现 | 思想对齐 |
| twin-Q TD3 critic | 已实现 | 对齐 |
| chunk TD target | 已实现 | 对齐 |
| BC regularizer `beta` | 已实现 | 对齐，并加了 warmup/online 分段 |
| reference action dropout | 已实现 | 对齐 |
| replay buffer | 已实现 | 基本对齐 |
| intervention 替换 `a_tilde` | 已实现 | 对齐 |
| 真人 intervention | 用 expert 模型模拟 | 机制近似，不是真人 |
| VLA warmup rollout | 未严格实现 | 主要差异 |
| stride=2 chunk subsampling | 未严格实现 | 次要差异 |
| Gaussian actor | 本地是 deterministic TD3 actor | 工程近似 |

## 14. 最重要的工程判断

如果现在 Stage2 一上来就掉成功率，最值得优先检查的是：

```text
warmup 阶段是否真的执行了 a_tilde？
```

因为论文 warmup 是行为侧 warmup，也就是让 base VLA 真的控制环境一段时间。本地当前更多是训练侧 warmup，也就是“不够数据不训 actor”和“loss 更保守”。这两者不完全等价。

更严格的论文实现应该是：

```text
前 N_warm 环境步：
    action = a_tilde
    存入 replay buffer

之后：
    action = actor(x, a_tilde)
    如果触发 intervention，则 action = a_expert，并且 a_tilde = a_expert
```
