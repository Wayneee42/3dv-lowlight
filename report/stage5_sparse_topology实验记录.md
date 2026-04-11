# Stage5 Sparse / Topology 实验记录

## 1. 文档用途
这份文档用于给后续新会话快速恢复 `stage5` 几何优化实验上下文，重点覆盖：

- 实验目标
- 已实现机制与代码方向
- 已跑实验及结论
- 当前最优可参考结果
- 待实现方案与不建议优先继续的方向

文档更新时间基于当前工作区状态，仓库根目录为 `D:\github\3DRR_low_light`。

## 2. 当前实验目标
当前主目标不是单纯提高 `PSNR/SSIM/LPIPS`，而是：

- 优化 `stage5` 几何
- 让 test 渲染出现更有说服力的几何变化
- 支撑后续“模块有效”的可视化对比

因此，判断标准应同时看：

- `results.json`
- `per_view.json`
- `sparse_signal_diagnostics.txt`
- 典型 test 视角中的边缘、空洞、遮挡、floaters 变化

## 3. 总体结论
到目前为止，结论已经比较清楚：

1. `stage5b` 的 sparse-guide 本身不是完全失效，而是信号太弱。
2. 仅靠 `Stage5 no-densify` 下的 sparse regularization，很难显著改变最终结果。
3. 真正能让图像出现明显变化的是 `stage5c_sparse_topology`，也就是让 sparse prior 参与 `spawn/prune`。
4. 目前 `stage5c` 已经能带来可感知几何变化，但平均指标仍然浮动，说明方法方向可能是对的，但还不够稳。
5. `depth prior` 在当前这条 `stage5` 几何实验线上不像主杠杆，可以先关闭，用于加快实验迭代。

## 4. 阶段演进概览

### 4.1 `stage5b_ft_on`
目标：

- 验证基础 sparse-guide 是否有效

结论：

- `meta_used=1` 只能说明 metadata 加载成功，不说明 sparse 信号强
- 通过新增的 `sparse_signal_diagnostics.txt` 确认 sparse loss 有非零梯度，但梯度占比很低
- 问题不在“完全没用上 sparse”，而在“它只是弱正则”

代码侧关键新增：

- `train.py` 中加入 `sparse_signal_diagnostics.txt`
- 记录 sparse loss、梯度占比、support score 分位数等

### 4.2 `stage5b_ft_v1`
实现：

- `hardest + random` 混合采样
- sparse weight 前强后弱 schedule

结论：

- 比 `stage5b_ft_on` 略强，但仍然没有把 sparse-guide 从“弱正则”提升到“真正能改结果的信号”
- 根因仍是覆盖率低、梯度占比低

### 4.3 `stage5b_ft_v2`
实现：

- `point_to_plane` sparse loss
- reliability-aware brightness / gradient 加权
- global hard mining
- 更强 sparse schedule

结论：

- 几何诊断量确实更强
- 但 `Stage5 no-densify` 下几何依旧很快进入平台期
- 最终指标变化依然很小

含义：

- 继续在 `stage5b` 上堆更复杂 sparse loss，边际已经不高

### 4.4 `stage5c_sparse_topology`
实现：

- 周期性 `spawn / prune`
- sparse prior 直接参与 topology 调整
- topology event 写入 `sparse_signal_diagnostics.txt`

这是目前最关键的一步，因为它把 sparse prior 从“拉已有 Gaussian”变成了“主动改拓扑”。

结论：

- 这是目前第一条真正能让 test 图出现明显几何变化的方向
- 但当前版本仍然偏 `prune-led`
- 平均指标只小幅改善或浮动

## 5. 当前工作区已实现内容

### 5.1 代码层面已实现
以下功能已经进入当前工作区代码：

- `stage5b_ft_v1`
- `stage5b_ft_v2`
- `stage5c_sparse_topology`
- sparse 诊断日志
- sparse `END_STEP`
- sparse 关闭后诊断梯度计算的安全保护
- `GLOBAL_MINING_REFRESH_INTERVAL=100`
- adaptive coverage
- `SEED_FEATURE_INIT` 已回退到 `nearest_gaussian_copy`
- `stage5c` 当前仓库配置为：
  - `DEPTH.ENABLED: false`
  - `PRUNE_START_STEP: 2000`
  - `SPARSE.END_STEP: 3000`
  - `WEIGHT_SCHEDULE: constant`
  - `WEIGHT=0.1`

注意：

- 当前仓库里的 `config/stage5c_sparse_topology/*.yaml` 已经被后续实验改动过，不一定等于“当前最优结果”的真实配置。
- 如果要恢复当前最优本地 run，优先看输出目录里的 `config.yaml`，而不是直接相信仓库配置。

### 5.2 当前仓库 `stage5c` 配置状态
当前文件 [config/stage5c_sparse_topology/laboratory.yaml](/D:/github/3DRR_low_light/config/stage5c_sparse_topology/laboratory.yaml) 的关键点：

- `DEPTH.ENABLED: false`
- `SPARSE.END_STEP: 3000`
- `SPARSE.WEIGHT=0.1`
- `SPARSE.WEIGHT_SCHEDULE=constant`
- `SPARSE_TOPOLOGY.PRUNE_START_STEP=2000`
- `SPARSE_TOPOLOGY.ADAPTIVE_COVERAGE_ENABLED=true`
- `SEED_FEATURE_INIT=nearest_gaussian_copy`

这代表仓库当前默认更接近“无 depth + 分段 prune”的实验分支，而不是“当前最佳本地结果”的完整复刻。

## 6. 当前最值得参考的输出目录

### 6.1 `stage5c_sparse_topology/Laboratory`
路径：

- [outputs/stage5c_sparse_topology/Laboratory](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory)

这是目前本地保留结果里最值得当作 `stage5c` 主参考的版本。

关键文件：

- [config.yaml](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/config.yaml)
- [step_5000/results.json](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/step_5000/results.json)
- [step_5000/per_view.json](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/step_5000/per_view.json)
- [sparse_signal_diagnostics.txt](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/sparse_signal_diagnostics.txt)

最终结果：

- `PSNR = 15.519516`
- `SSIM = 0.602373`
- `LPIPS = 0.399316`

相对 `stage5b_ft_off/Laboratory`：

- [off results](/D:/github/3DRR_low_light/outputs/stage5b_ft_off/Laboratory/step_5000/results.json)
- `PSNR +0.007703`
- `SSIM +0.000586`
- `LPIPS -0.001179`

诊断 summary：

- `final_num_gaussians = 355515`
- `total_prune_count = 25806`
- `total_spawn_count = 1388`
- `mean_coverage_hole_ratio = 0.03413`

解读：

- 这是一个明显的 `prune-led` 拓扑版
- 相比 `stage5b_ft_off`，图像变化更明显
- 指标提升不大，但已经能支撑“几何确实被改动”的结论

### 6.2 `stage5c_sparse_topology/Laboratory_no_depth_stage`
路径：

- [outputs/stage5c_sparse_topology/Laboratory_no_depth_stage](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory_no_depth_stage)

这是“关闭 depth prior + `500~2000 spawn-only, 2000~3000 再 prune`”的实验分支。

关键文件：

- [config.yaml](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory_no_depth_stage/config.yaml)
- [step_5000/results.json](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory_no_depth_stage/step_5000/results.json)
- [sparse_signal_diagnostics.txt](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory_no_depth_stage/sparse_signal_diagnostics.txt)

最终结果：

- `PSNR = 15.501121`
- `SSIM = 0.601376`
- `LPIPS = 0.400264`

相对当前主参考 `Laboratory`：

- 最终更差
- 但中途 `2000 / 3000` 的渲染结果曾更强

诊断 summary：

- `final_num_gaussians = 362193`
- `total_prune_count = 18652`
- `total_spawn_count = 912`
- `mean_coverage_hole_ratio = 0.02243`

解读：

- 这版说明“depth prior 不是 stage5 当前主杠杆”
- 同时也说明 delayed prune 让模型更厚、更稳，但最后清理不够，最终不如当前主参考

### 6.3 `stage5c_sparse_topology/GearWorks`
路径：

- [outputs/stage5c_sparse_topology/GearWorks](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/GearWorks)

最终结果：

- [stage5c results](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/GearWorks/step_5000/results.json)
- `PSNR = 14.214121`
- `SSIM = 0.579283`
- `LPIPS = 0.503271`

对照：

- [stage5b_ft_off results](/D:/github/3DRR_low_light/outputs/stage5b_ft_off/GearWorks/step_5000/results.json)
- `PSNR = 14.215268`
- `SSIM = 0.579284`
- `LPIPS = 0.503308`

解读：

- `GearWorks` 的平均指标目前基本持平
- 更适合拿来观察“是否有局部几何变化”，不适合作为主打量化证据

## 7. 已形成的稳定判断

### 7.1 关于 sparse 信号
- `meta_used=1` 只说明 `points_meta.npz` 被成功使用
- 不能说明 sparse 信号强
- sparse 信号是否真正起作用，要看：
  - `sparse_grad_ratio_means`
  - `sparse_distance_mean`
  - `support/quality` 分布
  - topology event 统计

### 7.2 关于 `stage5b`
- `stage5b_ft_on / v1 / v2` 已经基本证明：
  - sparse regularization 会起作用
  - 但在 `Stage5 no-densify` 框架里，它主要只是弱正则
  - 很难单独把几何结果拉开

### 7.3 关于 `stage5c`
- `stage5c` 才是真正有杠杆的方向
- 但当前版本偏 `prune-led`
- 视觉变化已经明显，但指标依然浮动

### 7.4 关于 `depth prior`
- 在当前 `stage5` 几何优化线上，`depth prior` 不是最关键因素
- 关掉它可以明显缩短训练时间
- 适合作为后续几何机制实验的默认快速设置

### 7.5 关于 `knn_weighted_copy`
- 曾尝试让 spawn 初始化从“复制最近单个 Gaussian”改为 `knn_weighted_copy`
- 结果并没有证明这个改动值得作为默认方案
- 当前默认已回退到 `nearest_gaussian_copy`

## 8. 可视化结论

### 8.1 `Laboratory`
相对 `stage5b_ft_off`，`Laboratory` 已经表现出“视觉更像变好，但平均指标提升有限”的典型模式。

已知 per-view 结论：

- `PSNR` 改善视角 `3/6`
- `SSIM` 改善视角 `6/6`
- `LPIPS` 改善视角 `5/6`

适合做可视化主案例。

### 8.2 `GearWorks`
目前更像“变化存在，但不稳定”。

- 不适合作为主证据
- 适合作为补充 case，用来看方法在复杂结构场景下是否能稳定工作

## 9. 当前最推荐的新实验基线
如果要继续做 `stage5` 几何优化，建议以“当前最优 `stage5c` 逻辑，但保留 `no depth prior`”为新基线，而不是继续沿用 `Laboratory_no_depth_stage` 的 delayed-prune 节奏。

更具体地说：

- 保留 `stage5c_sparse_topology` 的 topology 方向
- 保留 adaptive coverage
- 保留 `nearest_gaussian_copy`
- 关闭 `DEPTH.ENABLED`
- 不把 delayed prune 当作下一版主基线

原因：

- delayed prune 版中途更强，但最终更差
- 当前主参考 `Laboratory` 更适合当“方法主线”
- 后续应围绕“持续几何约束质量”继续强化，而不是继续大改节奏

## 10. 待实现主方案

### 10.1 主方案名称建议
建议新开一个新配置组，例如：

- `stage5c_vsurface`

不要直接覆盖当前 `stage5c_sparse_topology`，这样便于保留现有主参考结果。

### 10.2 主方案目标
把“局部表面方向”从：

- spawn 时的一次性初始化信息

升级成：

- `0~3000` sparse 有效阶段内的持续几何约束

目标是让 spawned / active Gaussians 在训练期间持续受到 sparse 局部表面法向和各向异性形状的约束，提升几何稳定性与可视化说服力。

### 10.3 建议实现内容
在 `SparsePointRegularizationLoss` 内新增两类持续几何项：

1. `orientation_alignment_loss`
- 用 sparse 邻域 PCA 得到 `sparse_normal`
- 用 Gaussian 当前 `quat` 推出 `gaussian_normal_axis`
- 约束 `1 - |dot(gaussian_normal_axis, sparse_normal)|`
- 只对 plane 稳定的样本生效

2. `anisotropic_scale_target_loss`
- 从 sparse 邻域估计 `local_sparse_radius`
- 构造：
  - `target_tangent_scale`
  - `target_normal_scale`
- 约束当前 Gaussian 的切向尺度、法向尺度接近上述目标
- 用鲁棒方式，不做硬性精确拟合

这两个项应直接并入当前 `SparsePointRegularizationLoss`，不要单独开新 loss module。

### 10.4 新增配置项建议
放到 `PRIORS.SPARSE` 下：

- `ORIENTATION_ENABLED: true`
- `ORIENTATION_WEIGHT: 0.05`
- `ANISOTROPIC_SCALE_TARGET_ENABLED: true`
- `ANISOTROPIC_SCALE_TARGET_WEIGHT: 0.05`
- `TANGENT_SCALE_RATIO: 0.9`
- `NORMAL_SCALE_RATIO: 0.24`
- `TARGET_TANGENT_SCALE_MIN: 0.005`
- `TARGET_TANGENT_SCALE_MAX: 0.05`
- `TARGET_NORMAL_SCALE_MIN: 0.002`
- `TARGET_NORMAL_SCALE_MAX: 0.02`

### 10.5 日志新增项建议
继续写入 `sparse_signal_diagnostics.txt`：

- `orientation_loss`
- `orientation_alignment_mean`
- `orientation_alignment_p50`
- `orientation_alignment_p90`
- `anisotropic_scale_target_loss`
- `target_tangent_scale_mean`
- `target_normal_scale_mean`
- `gaussian_tangent_scale_mean`
- `gaussian_normal_scale_mean`
- `stable_plane_ratio`

### 10.6 这版为什么值得做
当前主问题不是“所有 Gaussian 都学不动”，而是：

- sparse plane 的法向信息没有被持续利用
- spawned 点出生后缺少持续方向约束
- geometry 改动还不够稳

所以，比起先全局提高 `LR_MEANS / LR_SCALES / LR_QUATS`，更应该先把“持续局部表面方向约束”补上。

## 11. 暂不建议优先做的方向

### 11.1 全局提高 `means/scales` 学习率
理由：

- 风险是把旧几何一起扰动
- 当前更像是“约束质量不足”，不是“完全动不起来”

如果后续还要加强学习率，建议只做 newborn-only 局部增强，而不是全局抬高。

### 11.2 继续加大 `MAX_SPAWN_PER_EVENT`
理由：

- 目前问题不只是 spawn 数量
- 更关键的是：
  - coverage 判据
  - spawn 初始化质量
  - 持续几何约束是否足够

### 11.3 恢复 `knn_weighted_copy`
理由：

- 之前没有证明它对结果有明确帮助
- 复杂度增加了，但收益没有被证实

## 12. 新会话建议读取顺序
如果新会话要继续这条线，建议按下面顺序读取：

1. 本文档  
   [stage5_sparse_topology实验记录.md](/D:/github/3DRR_low_light/report/stage5_sparse_topology实验记录.md)
2. 当前主参考输出  
   [outputs/stage5c_sparse_topology/Laboratory/config.yaml](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/config.yaml)  
   [outputs/stage5c_sparse_topology/Laboratory/sparse_signal_diagnostics.txt](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory/sparse_signal_diagnostics.txt)
3. 对照输出  
   [outputs/stage5c_sparse_topology/Laboratory_no_depth_stage/config.yaml](/D:/github/3DRR_low_light/outputs/stage5c_sparse_topology/Laboratory_no_depth_stage/config.yaml)
4. 参考思路  
   [sparse-guide机制的参考修改思路.txt](/D:/github/3DRR_low_light/sparse-guide机制的参考修改思路.txt)
5. 关键实现文件  
   [train.py](/D:/github/3DRR_low_light/train.py)  
   [core/losses/modules.py](/D:/github/3DRR_low_light/core/losses/modules.py)  
   [core/losses/builder.py](/D:/github/3DRR_low_light/core/losses/builder.py)

## 13. 一句话状态总结
当前已经证明：

- `stage5b` 的 sparse-guide 太弱
- `stage5c` 的 topology 才是真正有杠杆的方向
- `Laboratory` 已经有可做可视化对比的基础
- 下一步最值得做的是“基于当前最优 `stage5c`，保留 no depth prior，引入持续局部表面方向约束”
