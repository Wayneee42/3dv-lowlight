#colmap sparse-guide相关

##1. 固定姿态 COLMAP sparse 导出

固定姿态 COLMAP 稀疏几何由 [build_fixed_pose_colmap_sparse_init.py](/D:/github/3DRR_low_light/tools/build_fixed_pose_colmap_sparse_init.py) 构建。该脚本不是重新估计相机位姿，而是直接读取官方 `transforms_train.json` 中的位姿，写入 manual model，再只执行：

1. feature extraction；
2. matching；
3. point triangulation；
4. point filtering。

脚本执行时还会先调用 [prepare_low_light_batch()](/D:/github/3DRR_low_light/core/libs/augment.py) 生成 deterministic 的 `supervision` 图像，再把这些增强后的训练图像送给 COLMAP。输出会写入场景目录下的 `auxiliaries/colmap_sparse/`，其中包括：

- `points.npy`
  - 过滤后的稀疏点坐标；
- `points_meta.npz`
  - 对应的 `track_len` 和 `reproj_error`；
- `report.json`
  - 预处理统计信息。

需要特别强调的是：

- `track_len` 与 `reproj_error` 在初始化阶段并不参与 sparse 点筛选；
- 这两项 metadata 是为 Stage 5 的 sparse-guided regularization 准备的；
- 当前主线中的“reliability-aware”主要发生在 Stage 5 的 sparse support weighting

## 2. Sparse-Guided Regularization

这是当前 Stage 5 的核心模块，实现位于 [SparsePointRegularizationLoss](/D:/github/3DRR_low_light/core/losses/modules.py)。

当前最优 Stage 5 配置相较于最初版本做了强化，代表设置包括：

- `SAMPLE_POINTS: 4096`
- `WEIGHT: 0.1`
- `MIN_OPACITY: 0.2`
- `KNN_K: 3`
- `ROBUST_SCALE: 0.05`
- `META_ENABLED: true`

其关键逻辑如下。

#### 5.3.1 Active Gaussian 选择

该 loss 不约束全部高斯，而是先基于：

- `sigmoid(opacity) > MIN_OPACITY`

筛出 active Gaussians。这样做的目的是只对真正参与成像的高斯施加 sparse support 约束。

#### 5.3.2 随机采样

在 active Gaussians 中，再随机采样 `SAMPLE_POINTS` 个点进入当前 step 的 sparse-guided 约束。这保证该项 loss 仍然是：

- 小批量；
- 弱监督；
- 不主导整体训练轨迹。

#### 5.3.3 kNN Barycenter

对每个 sampled Gaussian center，不是只约束到单个最近 sparse 点，而是查找最近 `KNN_K` 个 sparse points，并构造 support-weighted barycenter，再约束 Gaussian center 到这个 barycenter 的距离。这样可以减弱单个稀疏点离群时的剧烈拉扯。

#### 5.3.4 Robust Distance

当前实现用的是 Charbonnier 风格的鲁棒距离，而不是硬 `clamp`。这使 sparse-guided 对远距离点不会像普通 L2 那样过于激进。

#### 5.3.5 Quality / Density-Aware Support Score

当前主线的“reliability-aware”主要体现在这里。

Stage 4 预处理导出的 `points_meta.npz` 中包含：

- `track_len`
- `reproj_error`

在 Stage 5 训练时，这些元数据会被读入为：

- `colmap_sparse_track_len`
- `colmap_sparse_reproj_error`

在 loss 内部，当前 sparse support score 由以下因素构成：

1. `track_score`
   - 由 `log1p(track_len)` 构造并做中位数归一化；
2. `error_score`
   - 由 `exp(-reproj_error / error_scale)` 构造；
3. `quality_score`
   - `track_score * error_score`；
4. `density_score`
   - 由 sparse 点局部 kNN 半径得到；
5. `support_score`
   - `quality_score * density_score`。

最终 barycenter 权重不是纯 `1 / d`，而是：

- `support_score / distance`

归一化后的结果。

因此，当前 Stage 5 的 sparse-guided 不是简单最近邻拉拽，而是：

- quality-aware；
- density-aware；
- robust；
- training-time persistent geometric support。