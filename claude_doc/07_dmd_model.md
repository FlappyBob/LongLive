# DMD 模型：分布匹配蒸馏·generator_loss·critic_loss

> 核心文件：[model/dmd.py](../model/dmd.py)，[model/base.py](../model/base.py)，  
> [model/dmd_switch.py](../model/dmd_switch.py)

---

## 一、DMD 是什么

DMD（Distribution Matching Distillation，[论文](https://arxiv.org/abs/2311.18828)）是 LongLive 的训练损失框架。

```
核心思路：
  传统扩散模型需要 1000 步去噪，每步调用一次大模型
  DMD 用"分数匹配"方法，让少步（4步）的"学生模型"逼近多步的"教师分布"

LongLive 的 DMD 架构：
  ┌─────────────────────────┐
  │  generator（学生）       │  ← 训练目标，产生视频
  │  fake_score（评判器）    │  ← 评估"学生"生成有多像真实
  └─────────────────────────┘
  + real_score（教师）       ← 预训练的 Wan 模型，不更新参数
```

---

## 二、`SelfForcingModel` 基类

文件：[model/base.py](../model/base.py)

```python
class SelfForcingModel(nn.Module):
    def __init__(self, args, device):
        # generator: 学生模型（CausalWanModel，需要训练）
        self.generator = WanDiffusionWrapper(is_causal=args.causal, ...)
        
        # fake_score: 评判器（同架构但参数独立，用于 DMD 损失）
        self.fake_score = WanDiffusionWrapper(is_causal=False, ...)
        
        # text_encoder: UMT5（只在主进程初始化，其他 rank 靠 broadcast）
        self.text_encoder = WanTextEncoder()
        
        # vae: 编解码（eval 模式，不训练）
        self.vae = WanVAEWrapper()
        
        # scheduler
        self.scheduler = generator.get_scheduler()
```

---

## 三、DMD 前向传播

文件：[model/dmd.py](../model/dmd.py)

```
DMD.forward(noise, conditional_dict, unconditional_dict)
│
├── [Step 1] 生成视频（Self-Forcing 方式）
│   → inference_pipeline.generate(noise, cond)
│   → 得到 estimated_x0（学生模型生成的视频）
│
├── [Step 2] 计算 KL 散度梯度（评判质量）
│   → _compute_kl_grad(estimated_x0, cond, uncond)
│   → 得到 kl_grad（指导学生向真实分布移动的梯度方向）
│
├── [Step 3] 反向传播
│   → estimated_x0.backward(kl_grad)
│   → generator 的参数得到梯度更新
│
└── [Step 4] 更新 fake_score（对抗训练）
    → _compute_fake_score_loss(kl_grad, estimated_x0, cond)
```

---

## 四、`_compute_kl_grad`：核心损失

文件：[model/dmd.py:60-130](../model/dmd.py)

```python
def _compute_kl_grad(self, noisy_video, estimated_clean_video, timestep,
                     conditional_dict, unconditional_dict, normalization=True):
    """
    计算 KL 散度梯度（DMD 论文的公式 7）。
    
    直觉：
    - fake_score 评估"学生生成的带噪声视频在 timestep t 时"的得分
    - real_score（= 真实扩散模型）评估同一视频的得分
    - 梯度 = fake_score - real_score
      → 指导学生往"fake_score 低、real_score 高"的方向走
      → 即：让学生生成更像真实数据分布的视频
    """
    
    # Step 1: 从学生预测出的干净帧 x0 计算带噪声的 x_t
    # （随机采样一个训练用的 timestep）
    noisy_image_or_video = scheduler.add_noise(estimated_clean_video, noise, timestep)
    
    # Step 2: fake_score 对带噪声的学生输出打分
    # （fake_score 是一个扩散模型，它评估 x_t 在 timestep t 时的"干净程度"）
    with torch.no_grad():
        fake_flow_pred = fake_score.forward(
            noisy_image_or_video, conditional_dict, timestep
        )
        fake_pred_x0 = _convert_flow_pred_to_x0(fake_flow_pred, noisy_video, timestep)
    
    # Step 3: real_score（教师/基础 Wan 模型）对同一带噪声视频打分
    with torch.no_grad():
        real_flow_pred = self.real_score(...)  # 不参与训练
        real_pred_x0 = _convert_flow_pred_to_x0(real_flow_pred, noisy_video, timestep)
    
    # Step 4: 计算 KL 梯度
    # 公式：grad = (1/σ_t²) × (fake_x0 - real_x0)
    kl_grad = (fake_pred_x0 - real_pred_x0) / (sigma_t ** 2)
    
    # Step 5: 归一化（防止梯度过大）
    if normalization:
        kl_grad = kl_grad / kl_grad.abs().mean()
    
    return kl_grad, log_dict
```

**直觉理解**：

```
fake_score ≈ 学生自己的评判，带有"自我认知偏差"
real_score ≈ 真实数据分布的评判（更可靠）

fake_x0 - real_x0 = 学生认为"好"但真实认为"坏"的方向
  → 梯度指向"让学生不再犯这种错误"的方向

当 fake_x0 = real_x0 时，梯度为 0，训练收敛
```

---

## 五、fake_score 的更新

```python
def _compute_fake_score_loss(self, kl_grad, estimated_x0, timestep, cond):
    """
    fake_score 作为判别器，需要区分"学生生成的视频"（fake）
    和"在学生预测的 x0 上重新加噪得到的新 x_t"（也是 fake，但更接近真实）
    """
    # 计算 fake_score 的分类损失（类似 GAN 的 discriminator loss）
    fake_score_loss = F.mse_loss(fake_pred, target_pred)
    return fake_score_loss
```

fake_score 的更新使它更准确地评估学生模型的输出，从而提供更有用的梯度信号。

---

## 六、DMDSwitch（阶段二）

文件：[model/dmd_switch.py](../model/dmd_switch.py)

`DMDSwitch` 在 `DMD` 基础上增加了 prompt 切换训练：

```python
class DMDSwitch(DMD):
    def forward(self, noise, conditional_dict_list, switch_frame_indices):
        # conditional_dict_list: 多个 prompt 的嵌入列表
        # switch_frame_indices: 切换位置
        
        # 使用 StreamingSwitchTrainingPipeline 生成带切换的视频
        output = self.inference_pipeline.generate_with_switch(
            noise, conditional_dict_list, switch_frame_indices,
            requires_grad=True
        )
        
        # 计算多段的 KL 梯度（每段 prompt 用对应的 cond）
        for seg_idx, (start, end) in enumerate(segments):
            kl_grad_seg = self._compute_kl_grad(
                output[:, start:end],
                conditional_dict=conditional_dict_list[seg_idx],
                ...
            )
        ...
```

---

## 七、训练 timestep 采样策略

```python
# model/dmd.py:40-53
self.min_step = int(0.02 * num_train_timestep)   # = 20
self.max_step = int(0.98 * num_train_timestep)   # = 980

# ts_schedule: 训练初期用大 t（高噪声），后期用小 t（细节）
if self.ts_schedule:
    # 用 cosine schedule 在训练过程中逐渐减小 timestep 上界
    max_t = cosine_decay(self.max_step, self.current_step)
else:
    max_t = self.max_step

timestep = torch.randint(self.min_step, max_t, [B], device=device)
```

**timestep 为什么要是随机的？**

每个训练步骤的 `timestep` 是随机采样的（而不是固定的）。这样模型在所有噪声水平（从低噪到高噪）上都得到训练信号，避免过拟合到特定噪声级别。

---

## 八、guidance scale

```python
# model/dmd.py:44-49
self.real_guidance_scale = args.real_guidance_scale   # 通常 1.0~7.5
self.fake_guidance_scale = args.fake_guidance_scale   # 通常 0.0

# Classifier-Free Guidance:
# score_guided = score(cond) + scale × (score(cond) - score(uncond))
```

`real_guidance_scale > 1` 时，real_score 使用 CFG，让梯度信号更强（真实分布的评判更清晰）。
`fake_guidance_scale = 0` 时，fake_score 不使用 CFG（避免训练不稳定）。

---

## 九、完整训练损失汇总

```
total_loss = generator_loss + fake_score_loss

generator_loss:
  - 主要：estimated_x0.backward(kl_grad)（DMD 分布匹配）
  - 可选：+ flow_matching_loss（监督损失，防止模式崩溃）

fake_score_loss:
  - fake_score 的分类误差
  - 用于让 fake_score 更准确地评估学生输出
```

**注意**：`generator_loss` 不是一个标准的 PyTorch loss（因为梯度方向是手动指定的），而是通过 `estimated_x0.backward(kl_grad)` 直接把 KL 梯度"注入"计算图。
