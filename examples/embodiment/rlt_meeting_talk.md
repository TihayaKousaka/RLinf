# RLT 阶段性汇报讲稿

## 0. 一句话版本

今天主要汇报 RLT 在 ManiSkill PegInsertionSide joint control 上的接入和调试进展。RLT 的核心思路是：不直接 RL 微调整个 VLA，而是冻结 base VLA，用 RL token 抽一个紧凑状态表示，再训练一个很小的 actor-critic 在 VLA 给出的参考动作附近做在线强化学习微调。我们目前已经把 Stage2 的主要结构接进 RLinf，包括 frozen VLA、frozen RL token、residual actor、twin-Q critic、replay buffer、reference action dropout 和 expert intervention；现在主要问题是 eval success 没有明显提升，甚至会下降，初步判断和论文里的显式 VLA warmup、intervention 数据吸收、critic 引导 actor 偏离 base VLA 这几块还没有完全对齐有关。

## 1. 背景：为什么要做 RLT

可以这么开场：

> 我们现在的问题是，base VLA 通过 SFT 已经能完成一部分 PegInsertionSide，但在接触、插入这种精细阶段，成功率和稳定性还不够。直接对整个 VLA 做 RL 成本很高，显存、采样效率和训练稳定性都不好。所以 RLT 的目标是保留 VLA 的先验能力，只训练一个轻量的 RL head，让它在 VLA 已经给出的动作附近做局部修正。

RLT 解决的是一个很实际的问题：

```text
VLA 有强先验，但动作不够精细；
传统 RL 可以探索改进，但从零开始太慢；
RLT 把两者结合起来：VLA 给方向，RL 负责局部优化。
```

论文里的 Stage2 不是训练整个大模型，而是：

```text
冻结 VLA
冻结 RL token
训练小 actor
训练小 critic
用 replay buffer 做 off-policy TD3
```

也就是说，我们不是让大 VLA 在 RL 里反向传播，而是把 VLA 当成 perception backbone 和行为 prior。

## 2. RLT 整体算法在干什么

可以口播成：

> RLT 分两阶段。Stage1 是训练 RL token，让它从 VLA 的内部 embedding 里压缩出一个适合 RL 使用的状态向量。Stage2 是在线 RL 微调阶段，冻结 VLA 和 RL token，只训练 actor-critic。每个 rollout step，VLA 先给一个参考动作 chunk，RL token 给一个状态表示，然后 actor 根据这个状态和参考动作输出最终动作。critic 从 replay buffer 里学这个动作 chunk 的价值，actor 再根据 critic 的 Q 值去改进动作，同时用 BC regularizer 约束它不要离 VLA reference 太远。

用符号说：

```text
s: 当前观测，包括图像和机器人状态
l: 语言指令
z_rl: RL token 输出的紧凑状态表示
s_p: proprioception，本体状态
x = [z_rl, s_p]: actor/critic 真正使用的状态
a_tilde: base VLA 给的参考动作 chunk
a: actor 最终输出并执行的动作 chunk
Q(x, a): critic 预测这个动作 chunk 的价值
```

本地 actor 的形式是 residual：

```text
a = a_tilde + residual_actor(x, a_tilde)
```

这个设计的好处是：训练初始时 residual 被 zero init，所以一开始行为接近 base VLA，不是随机探索。

## 3. Stage2 训练循环

可以这么讲：

> Stage2 的训练循环是典型 off-policy actor-critic。rollout 采集 transition，存入 replay buffer；critic 从 buffer 采样，做 TD target 学 Q；actor 每隔几个 critic update 更新一次，通过最大化 Q 来改进动作，同时用 BC loss 贴近 VLA reference。

本地当前配置：

```yaml
num_action_chunks: 10
action_dim: 8
embedding_dim: 2048
proprio_dim: 9
critic_updates_per_actor: 2
buffer_capacity: 200000
min_buffer_size: 600
```

解释：

```text
actor 一次输出 10 步 joint delta 动作；
critic 输入的是一个完整 action chunk；
每次 training update 都训练 critic；
每 2 次 update 才训练一次 actor；
buffer 里至少有 600 条 transition 才开始训练。
```

actor loss 当前是：

```text
L_actor = bc_weight * MSE(a, a_tilde) - q_weight * Q(x, a) + delta_weight * delta_loss
```

白话解释：

```text
bc_weight: 让 actor 不要离 base VLA 太远；
q_weight: 让 actor 根据 critic 的评价去追求更高成功率；
delta_weight: 约束 chunk 内动作变化，目前 joint 配置里是 0。
```

当前权重：

```yaml
warmup_bc_weight: 7.0
warmup_q_weight: 0.05
online_bc_weight: 3.0
online_q_weight: 0.1
actor_warmup_updates: 200
actor_loss_ramp_updates: 400
```

也就是说：

```text
前 200 个 training update 很保守；
之后 400 个 update 从 warmup 权重线性过渡到 online 权重；
约 600 个 update 后才完全进入 online 权重。
```

## 4. 我们在 RLinf 里已经实现了什么

这一段可以直接作为“进展”讲：

> 目前 Stage2 的主体已经接通了。我们新增了 RLTStage2Policy，用来包装 frozen OpenPI VLA、frozen RL token、residual actor 和 twin-Q critic。rollout worker 会在环境交互时提取 `x` 和 `a_tilde`，并把它们随 trajectory 一起缓存下来。actor worker 收到 trajectory 后，会把它转成 Stage2 replay buffer 里的 transition，然后用 TD3 方式更新 critic 和 actor。

已完成的点：

```text
1. RLT Stage2 policy 接入 RLinf model system。
2. 支持 frozen VLA 提取 reference action chunk。
3. 支持 frozen RL token 提取 z_rl。
4. 实现 residual actor：a = a_tilde + residual。
5. 实现 twin-Q critic 和 target critic。
6. 实现 chunk-level TD target。
7. 实现 Stage2 replay buffer。
8. 实现 actor/critic 分 optimizer 训练。
9. 实现 actor-only train model，训练 worker 不加载 VLA/RL token，节省显存。
10. rollout 只同步 actor 权重，避免同步 VLA/critic。
11. 实现 reference action dropout。
12. 实现 expert intervention，并且 intervention 时会把 replay buffer 里的 a_tilde 替换成 expert action。
13. 写了 joint-control 的 Stage2 YAML。
14. 写了算法对齐文档，逐项对齐论文和本地实现。
```

对应文件：

```text
rlinf/models/embodiment/rlt_stage2/rlt_stage2_policy.py
rlinf/models/embodiment/rlt_stage2/components.py
rlinf/models/embodiment/rlt_stage2/replay_buffer.py
rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py
rlinf/workers/rollout/hf/huggingface_worker.py
rlinf/runners/embodied_runner.py
examples/embodiment/config/rlt_stage2_maniskill_joint.yaml
examples/embodiment/rlt_stage2_algorithm_alignment.md
```

## 5. 人在环和 expert intervention 是怎么做的

可以这样讲：

> 论文里 RLT 的 real robot 训练包含 human intervention。人可以在机器人执行失败趋势明显的时候接管，给出 teleoperation action。一个关键细节是：如果人接管，不只是执行人的动作，还要把 replay buffer 里的 reference action 也替换成人的动作。这样 actor 后面做 BC regularizer 的时候，是被拉向纠偏动作，而不是继续拉回错误的 base VLA 动作。

我们在仿真里没有真人接管，所以用一个更强的 expert VLA 来模拟 human intervention：

```text
base/student VLA: global_step_2000 actor
expert VLA: global_step_8000 actor
```

触发逻辑：

```yaml
intervention:
  enable: True
  success_metric: "env/success_once"
  baseline_warmup_steps: 10
  relative_drop: 0.10
  probability: 0.9
```

意思是：

```text
先收集前 10 个 rollout 的 success baseline；
后面如果 success 比 baseline 低 10% 以上，就允许 expert 介入；
允许介入时，有 0.9 概率用 expert action 替换学生 action。
```

本地已经对齐的关键点：

```text
intervention 时 action = expert_action；
同时 forward_inputs["a_tilde"] = expert_action；
并记录 intervention_flags。
```

这一点和论文是对齐的，因为论文要求 intervention 替换 VLA reference。

## 6. 当前实验现象

可以这样汇报：

> 目前已经能跑 Stage2，但 eval success 还没有明显提升。训练中有几个比较稳定的现象：第一，train 的 step0 `env/success_once` 往往很高，但 eval 并不一定高；第二，训练后 eval 没有持续提升，有时会下降；第三，critic loss 会逐渐下降，说明 critic 对 replay buffer 里的 TD target 在拟合；第四，actor 的 residual_abs_mean 一直在涨，说明 actor 确实在偏离 base VLA；但是这种偏离没有转化成 eval success 的提升。

对这些现象的解释：

```text
1. step0 env/success_once 高，不一定说明模型真的强。
   train env 是 auto_reset，而且只统计已经 done 的 episode；
   成功 episode 通常更短，失败 episode 可能还没 done；
   所以 train step0 会偏乐观。

2. eval 才是更可信的学生策略指标。
   eval 不使用 expert intervention；
   eval 是 deterministic student actor；
   所以 eval 下降说明学生 actor 自己没有学好。

3. critic_loss 下降不等于 actor 学到了好策略。
   critic 可以很好拟合 replay buffer；
   但 actor 一旦偏离 buffer 分布，Q 可能给出错误引导。

4. residual_abs_mean 涨但 eval 不涨，说明 actor 确实在动，但动的方向可能不是成功率提升方向。
```

当前我们主要看这些 W&B 指标：

```text
eval/success_once
env/success_once
env/num_trajectories
train/actor/residual_abs_mean
train/actor/bc_loss
train/actor/q_mean
train/critic/q1_mean
train/critic/q2_mean
train/rlt_stage2/critic_loss
train/replay_buffer/intervention_rate
intervention/enabled_next_rollout
```

## 7. 目前怀疑还没有完全对齐论文的地方

这一段要讲得稍微谨慎：

> 主体结构已经对齐，但有几处实现和论文还有差异，这些差异可能解释为什么 eval 没明显提升。

### 7.1 显式 VLA warmup 还没有严格实现

论文 warmup 是行为侧 warmup：

```text
前 N_warm 环境步：
  action = a_tilde
  把 base VLA 数据存进 replay buffer
```

本地目前主要是训练侧和 loss 侧 warmup：

```text
buffer 少于 min_buffer_size 不训练；
前 actor_warmup_updates 使用高 BC、低 Q；
residual actor zero-init，让初始动作接近 a_tilde。
```

这两者不是完全等价。

差别是：

```text
论文：warmup 阶段环境里一定执行 base VLA。
本地：warmup 阶段只是希望 actor 接近 base VLA，但 rollout 仍可能执行 actor(x,a_tilde)。
```

如果 actor 很早被 critic 拉偏，就可能污染 replay buffer，导致后续 eval 下降。

### 7.2 expert 能救 train，但 eval 只看学生

train 里 expert 介入后，`env/success_once` 可能被托起来；但 eval 里不走 expert，所以如果学生没有吸收 expert 数据，`eval/success_once` 不会提升。

所以要区分：

```text
train env success 提升：可能是 expert 救了；
eval success 提升：才说明学生学会了。
```

### 7.3 reference action dropout 的效果可能偏弱

论文说随机把 `a_tilde` 置零再传给 actor，避免 actor 只复制 reference。

本地 residual actor 是：

```text
a = a_tilde + residual_actor(x, dropped_or_normal_a_tilde)
```

也就是说 dropout 只影响 residual branch 的输入，但最终还是会把原始 `a_tilde` 加回来。这不一定错，但和论文里“让 actor 保持独立动作生成路径”的效果不是完全等价，可能弱一些。

### 7.4 replay buffer 采样窗口没有真正用上

YAML 里有：

```yaml
sample_window_size: 10000
```

但当前 Stage2 replay buffer 是自己实现的 uniform sample，基本没有用这个 sample window。也就是说如果前期有很多坏数据，它们会长期留在 buffer 里参与训练。

## 8. 对当前结果的判断

可以这样说：

> 现在的状态不是“完全没跑通”，而是“系统已经能训练，但还没有达到论文式稳定改进”。critic loss 下降说明 critic 在拟合数据；residual_abs_mean 上涨说明 actor 在更新；但是 eval 没提升说明 actor 的偏离没有变成真实成功率收益。这个更像是训练闭环稳定性问题，而不是单纯代码没跑起来。

换句话说：

```text
已经实现了 Stage2 的骨架；
现在卡在如何让 actor 的 residual 变成有效改进，而不是 Q exploitation 或偏离 base VLA。
```

## 9. 下一步计划

建议会议上把计划说成三层，从最确定到最探索：

### 9.1 先补论文式显式 VLA warmup

目标：

```text
前 N_warm rollout step 强制 action = a_tilde；
同时正常缓存 x、a_tilde、reward、next_x；
等 buffer 里有稳定 base VLA 数据后，再让 actor 控制。
```

这一步是最贴近论文的修正，也最可能提升训练稳定性。

### 9.2 调整 actor 更新节奏和 loss 权重

当前比较保守：

```yaml
actor_warmup_updates: 200
actor_loss_ramp_updates: 400
online_bc_weight: 3.0
online_q_weight: 0.1
```

候选实验：

```yaml
actor_warmup_updates: 100
actor_loss_ramp_updates: 200
```

如果 residual 太小、不动，可以适当降低 `online_bc_weight` 或提高 `online_q_weight`；如果 residual 变大但 eval 下降，则不应该继续提高 Q，而是先稳 critic 或提高 BC。

### 9.3 改 replay buffer 采样策略

可以考虑：

```text
减小 buffer_capacity 到 50000 或 100000；
或者实现 recent sample window；
或者提高 intervention transition 的采样权重。
```

这样避免早期坏数据长期稀释后面的好数据。

### 9.4 更细地诊断 expert 数据是否被学生吸收

重点看：

```text
train/replay_buffer/intervention_rate
actor/bc_loss on expert transitions
eval/success_once
```

如果 intervention_rate 很高但 eval 不涨，就说明 expert 数据进了 buffer，但学生没有学到。

## 10. 可以用于会议的 2 分钟总结

> 目前我们已经把 RLT Stage2 的主体接到了 RLinf 上。实现上包括 frozen VLA、RL token、residual actor、twin-Q critic、TD3 target、replay buffer、reference action dropout 和 expert intervention。算法上，actor 不是从零输出动作，而是在 base VLA 的参考动作 `a_tilde` 上预测 residual；critic 用 chunk-level TD target 学动作 chunk 的价值；actor loss 同时包含 Q 最大化和 BC regularizer，保证它既能改进动作，又不会离 VLA 太远。
>
> 现在实验上能正常跑起来，critic loss 会下降，actor residual 也确实在增长，说明训练链路是通的。但 eval success 没有明显提升，说明学生 actor 的偏离还没有转化成真实成功率提升。我们怀疑主要有三点：第一，论文里的显式 VLA warmup 还没有严格实现，我们现在更多是 loss warmup，不能保证 rollout 阶段一定执行 base VLA；第二，expert intervention 能救 train，但 eval 只测学生，如果学生没吸收 expert 数据，eval 不会涨；第三，critic 虽然在拟合 buffer，但可能给 actor 的改进方向还不可靠，导致 residual 变大但成功率不升。
>
> 下一步我计划先补显式 rollout warmup，让前 `N_warm` 步直接执行 `a_tilde`，保证 replay buffer 起始数据分布干净；然后再调 actor warmup/ramp 和 BC-Q 权重，最后看是否需要 recent replay 或 intervention 数据加权采样。

## 11. 可能被问到的问题

### Q1：为什么不用直接 fine-tune VLA？

因为直接 RL 更新大 VLA 成本高、慢、不稳定。RLT 的核心优势是只训练小 actor-critic，但仍然利用 VLA 的视觉语言表示和动作先验。

### Q2：为什么 actor 要看 `a_tilde`？

因为 `a_tilde` 是 VLA 的动作建议。actor 看它以后，学习空间从“从零生成动作”变成“在 VLA 建议附近做修正”，采样效率更高。

### Q3：为什么需要 BC regularizer？

因为 critic 早期不准，actor 如果只追 Q，容易被错误 Q 值拉到奇怪动作。BC regularizer 让 actor 不要离 base VLA 太远。

### Q4：为什么需要 reference action dropout？

因为 actor 如果总能看到 `a_tilde`，又被 BC 拉向 `a_tilde`，它可能只学复制。dropout 让 actor 在部分 batch 中看不到 reference，逼它学习一条独立动作生成路径。

### Q5：为什么 train success 和 eval success 不一致？

train 有 auto-reset，而且 expert intervention 可能接管；eval 是固定环境评估，并且只测学生策略。所以 eval 更能代表学生是否真的学会。

### Q6：现在最大的问题是什么？

不是训练链路没通，而是 actor residual 的增长没有带来 eval 成功率提升。也就是 actor 在偏离 base VLA，但偏离方向还不稳定或不正确。

### Q7：最优先改什么？

最优先补论文式显式 VLA warmup。因为这是算法层面对齐问题，比继续微调 `bc_weight` 更根本。

