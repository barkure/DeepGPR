# Deepwave 与 DeepGPR：PML 材料梯度、边界回流及吸收系数

核查日期：2026-10-11。核查官方 Deepwave **v0.0.26**，固定 commit `7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084`。官方源码下载在 `/tmp/deepwave-pml-review-v0.0.26`。本次为源码审阅及隔离 padding 函数的小实验，未安装或运行完整 Deepwave 传播器，未修改 DeepGPR 生产代码。

## 结论

Deepwave 会计算 **PML 区域的材料梯度**，并将复制材料产生的梯度求和回原模型边界。PML 材料由模型派生，不作为独立优化参数。这与“更新整个物理模型，PML 随模型延拓而不独立更新”的参数化相符。

但 Deepwave 没有对所有依赖模型的吸收系数计算总导数：其 `max_vel` 是普通数值，原生 scalar backward 不返回 `pml_profiles` 的梯度。官方梯度测试固定 `max_vel`，从而在扰动模型时保持吸收配置和内部时间步一致。

## 1. 复制延拓保持可微

`scalar.py` 调用公共预处理时指定 `['replicate']`；`common.extract_models` 在传播器自定义 Autograd Function 外直接使用 `torch.nn.functional.pad`，未替换其标准 backward。因此模型边界的一个值复制到多个 PML 点后，所有对应点的梯度都会加回该边界值。

来源：[scalar 预处理](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.py#L481-L498)、[复制 padding](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/common.py#L2071-L2087)。

若边界参数为 m，复制它的 PML 点集合为 P，则材料路径的梯度为：

`dL/dm = 原模型该点的材料敏感度 + Σ(P 中各点的材料敏感度)`。

这不意味着独立优化 PML：一次模型更新后，延拓值仍由新的边界值重新复制生成。

## 2. 原生核保留 PML 区域的材料敏感度

Scalar CPU 正反向遍历低侧 PML、内区、高侧 PML。正演保存的材料导数 history 含 PML 辅助项，反演在这些区域均累积速度梯度。CUDA 也只排除最外面的有限差分 halo，没有排除正在演化的 PML 点。

Python 反演按照 padded 模型的完整形状分配并返回材料梯度，之后再由 padding 的 backward 汇回模型。

来源：[CPU 正演](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.c#L307-L395)、[CPU 反演](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.c#L607-L690)、[CUDA](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.cu#L226-L377)、[完整材料梯度分配](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.py#L1144-L1149)、[返回梯度](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.py#L1318-L1333)。

Elastic 使用 CPML，其 λ、μ、浮力材料梯度同样覆盖 PML；正演材料 history 包含 CPML 修正，交错网格上的平均、倒数转换保留在可微图中。这可以借鉴组织方式，但声学/弹性方程的导数公式不能直接复制到电磁 Yee 网格。

来源：[Elastic CPU](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/elastic.c#L1933-L1994)、[Elastic CUDA](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/elastic.cu#L1419-L1494)、[可微物性转换](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/elastic.py#L23-L124)。

## 3. 吸收配置采用固定数值处理

`common.setup_pml` 的吸收强度由普通数值 `max_vel`、PML 厚度、网格间距、频率、dt 等构造。`scalar.py` 在未提供 `max_vel` 时使用 `v.abs().max().item()`，这条路径不参与 Autograd；其原生 backward 对 `pml_profiles` 返回 `None`。

官方测试包装器 `scalarprop` 明确固定 `max_vel=2000`，注释说明用于实际最大速度变化时保持一致。文档也说明可显式指定该参数，使不同模型的运行配置保持一致。

来源：[自动最大速度取值](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.py#L443-L458)、[profile 构造](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/common.py#L2486-L2554)、[profile 无梯度返回](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/src/deepwave/scalar.py#L985-L996)、[固定配置的测试](https://github.com/ar4/deepwave/blob/7dbaeda5fdfbff2a6d579dcc9f7f18488c5e6084/tests/test_scalar.py#L52-L55)、[官方参数说明](https://ausargeo.com/deepwave/usage#deepwave.scalar)。

**限制与推论：**如果每次使用 `max_vel=None`，扰动实际最大速度会使重建的吸收系数变化，而这条变化不在 native backward 中。源码因此不能支持“Deepwave 默认调用对模型关联的所有数值配置都有完整总导数”的结论。本次未对这一默认路径做传播数值验证。

## 4. 为什么它承认边缘梯度可能很大

官方专门有 [Large gradients at edges](https://ausargeo.com/deepwave/example_large_edge_gradient) 示例：波可能经过 PML，在充分衰减之前仍影响接收器；PML 内材料改变就会改变数据。将该区域敏感度汇回被复制的模型边界后，可以出现一格厚的大梯度。

这里“大梯度”可以是当前离散目标函数的正确导数，不等于梯度实现错误。官方给出增加模型缓冲区、可微平滑参数化等办法。示例还给出把上边界 PML 厚度设为零的反射边界办法，但这会改变物理边界条件，不能直接照搬到需要吸收边界的 GPR 模拟。

## 5. 与当前 DeepGPR 的对照

| 路径 | Deepwave v0.0.26 | 当前 DeepGPR |
|---|---|---|
| 材料延拓 | replicate padding，标准 backward 汇总复制点敏感度 | `_ExtendModel.backward` 只裁原模型区域 |
| 原生材料梯度 | 包含演化的 PML 区域 | CPU/CUDA 材料梯度核排除 PML cells |
| 吸收系数 | 不对 profile 反传；测试固定 `max_vel` | 每次按当前边界 ε/μ 平均值重建，并 detach |

DeepGPR 对照位置：[模型延拓](/home/ll/下载/DeepGPR-dev/src/DeepGPR/preprocessing/model_setup.py:72)、[CPU 材料梯度核](/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr_cpu.c:1652)、[CUDA 材料梯度核](/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr.cu:3041)、[CPML 系数构造](/home/ll/下载/DeepGPR-dev/src/DeepGPR/solver/pml.py:225)。

保持当前正演行为的修复，应同时补齐：PML 区域 ε/σ 的材料敏感度、复制延拓的转置求和、现有材料依赖 CPML 系数的 VJP。仅更换 padding backward 或仅去掉 `detach` 都不足以修复整个链路。

另一种设计是明确采用整个反演期间固定的吸收配置，类似 Deepwave 固定 `max_vel` 的做法；但必须让正演和 FD 都真正使用相同固定配置。这会改变当前“随模型重建系数”的数值函数，不能称作保持现有正演的梯度修复。

## 6. 隔离 padding 实验

直接从两项目源码 AST 中抽出延拓函数运行，使用 PyTorch 1.11.0、float64、3×3 模型、四边各复制 1 格，以“延拓材料所有值之和”为目标。**这只验证 padding，不是完整波场或损失测试。**

两种正演输出逐位相同。左上角物理参数复制成 4 个点，其有限差分均约为 4：

| 方法 | 左上角 AD | 左上角 FD |
|---|---:|---:|
| Deepwave 的标准 padding | 4 | 4.0000000000048885 |
| DeepGPR 当前 `_ExtendModel` | 1 | 4.0000000000048885 |

Deepwave 的整个梯度为 `[[4,2,4],[2,1,2],[4,2,4]]`，DeepGPR 当前裁剪 backward 为全 1。该实验具体说明复制回流差别；此前 DeepGPR 波场审计已单独验证其实际影响。

实验数据：[padding_probe.json](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/deepwave_comparison_20261011/padding_probe.json)。
