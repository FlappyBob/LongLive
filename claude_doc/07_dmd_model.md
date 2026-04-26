# DMD — 分布匹配蒸馏损失

**文件**：[model/dmd.py](../model/dmd.py)  
**基类**：[model/base.py](../model/base.py)

---

## 一、定位

`DMD`（Distribution Matching Distillation）是训练的**损失计算核心**，实现了从 14B 教师模型向 1.3B 学生模型的知识蒸馏。

架构继承关系：
```
BaseModel (model/base.py)
└── SelfForcingModel
    └── DMD (model/dmd.py)
        └── DMDSwitch (model/dmd_switch.py)  ← 增加 prompt 切换训练
```

---

## 二、BaseModel 初始化

```python
BaseModel._initialize_models(args, device):
    self.generator   = WanDiffusionWrapper(is_causal=True,  ...)   # 待训练，grad=True
    self.real_score  = WanDiffusionWrapper(is_causal=False, real_name)  # 教师，grad=False
    self.fake_score  = WanDiffusionWrapper(is_causal=False, fake_name)  # Critic，grad=True
    self.text_encoder = WanTextEncoder()   # grad=False
    self.vae          = WanVAEWrapper()    # grad=False
```

三个 Diffusion 模型的角色：

| 模型 | 参数量 | 梯度 | 作用 |
|------|--------|------|------|
| `generator` | 1.3B（因果） | ✓ | 被训练的生成器 |
| `real_score` | 14B（非因果） | ✗ | 教师模型，提供真实分布的梯度指导 |
| `fake_score` | 1.3B（非因果） | ✓ | Critic，拟合生成器的输出分布 |

---

## 三、DMD 损失原理

基于论文 [DMD2: arxiv/2311.18828]。

### KL 梯度（eq.7）

```
给定生成器输出 x0（fake video），在随机 timestep t 加噪得到 xt

fake_score(xt | prompt) → pred_fake_x0    ← Critic 认为 xt 来自哪里
real_score(xt | prompt) → pred_real_x0    ← 教师认为 xt 来自哪里

KL梯度 = pred_fake_x0 - pred_real_x0

直觉：如果 fake > real，说明 Critic 认为这个位置更像生成物而非真实数据
      → Generator 需要往 real 方向优化
```

### 梯度归一化（eq.8）

```python
p_real = (x0_estimated - pred_real_x0)  # 真实 score 的"残差"
normalizer = abs(p_real).mean(dim=[1,2,3,4], keepdim=True)
grad = grad / normalizer                  # 防止梯度量级随 timestep 变化
```

### DMD Loss

```python
dmd_loss = 0.5 × MSE(x0, (x0 - grad).detach())
```

这等价于让 `x0` 朝 `x0 - grad` 方向移动（伪目标，梯度已 detach），实际梯度是 `grad` 本身。

---

## 四、generator_loss()

```python
generator_loss(image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, initial_latent)
    → (dmd_loss, log_dict)
```

步骤：
```
1. _run_generator:
   SelfForcingTrainingPipeline.inference_with_trajectory(noise, ...)
   → pred_image [B, F, C, H, W]（fake 视频）
   → gradient_mask（哪些帧需要梯度）
   → timestep_from, timestep_to（去噪步骤范围）

2. compute_distribution_matching_loss(pred_image, ...):
   → 随机采样 timestep（基于 ts_schedule 限制范围）
   → add_noise 得到 noisy_latent
   → _compute_kl_grad → grad
   → dmd_loss = 0.5 × MSE(pred_image, pred_image - grad)
```

### Timestep 调度（ts_schedule）

```python
min_timestep = denoised_timestep_to   if ts_schedule else 0
max_timestep = denoised_timestep_from if ts_schedule_max else 1000
```

如果 Generator 当前从 t=750 去噪到 t=500，则 DMD loss 的 timestep 也限制在 [500, 750] 范围内，聚焦在最相关的噪声级别。

---

## 五、critic_loss()

```python
critic_loss(image_or_video_shape, ...)
    → (denoising_loss, log_dict)
```

步骤：
```
1. with torch.no_grad(): generator unroll → generated_image（fake）
2. 随机 timestep 加噪 → noisy_generated
3. fake_score(noisy_generated) → pred_fake_x0
4. denoising_loss = MSE(generated_image, pred_fake_x0)
   （训练 Critic 去拟合生成器的分布）
```

Critic 的损失是普通的去噪 MSE，让 Critic 学会"认识"生成器产生的图像分布。

---

## 六、训练交替策略

在 `Trainer.train()` 中：

```python
TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)
# dfake_gen_update_ratio = 5，即每 5 步更新一次 Generator，每步更新一次 Critic
```

```
step 0: critic_loss → critic.backward() → critic.step()
step 1: critic_loss → ...
step 2: critic_loss → ...
step 3: critic_loss → ...
step 4: critic_loss → ...
step 5: generator_loss + critic_loss → generator.step() + critic.step()
step 6: critic_loss → ...
...
```

这种安排让 Critic 比 Generator 更新更频繁，保证 Critic 能紧跟 Generator 的分布变化。

---

## 七、DMDSwitch 扩展

`DMDSwitch`（[model/dmd_switch.py](../model/dmd_switch.py)）在 `DMD` 基础上增加了 prompt 切换训练支持：

- `generator_loss` 中使用 `switch_conditional_dict` 和 `switch_frame_index`
- 生成时在 `switch_frame_index` 处切换 prompt（通过 `SwitchCausalInferencePipeline`）
- 让 Generator 学会在切换 prompt 时保持视觉连贯性

---

## 八、StreamingTrainingModel

`model/streaming_training.py` 中的 `StreamingTrainingModel` 是阶段二训练的状态管理器：

```python
class StreamingTrainingModel:
    def setup_sequence(conditional_dict, max_length=240, ...)
    def generate_next_chunk(requires_grad) → (chunk, chunk_info)
    def compute_generator_loss(chunk, chunk_info) → (loss, log_dict)
    def compute_critic_loss(chunk, chunk_info) → (loss, log_dict)
    def can_generate_more() → bool
```

它将长序列（240 帧）拆分为 21 帧的 chunk，每次 `generate_next_chunk` 推进一个 chunk，KV Cache 在 chunk 之间持续保持，实现了对显存的精细控制。
