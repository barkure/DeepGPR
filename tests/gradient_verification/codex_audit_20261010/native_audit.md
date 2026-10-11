# 原生 Yee/CPML 梯度审计与最小修复

审计日期：2026-10-10。起点为 `test/gradient-verification` 的 `3bfc1e998851dbe489cf3a56b3a7cb27d8f476ba` 及当时未提交的梯度验证文件。原有 t01–t07 和结果未被本审计覆盖。所有数据均为合成模型；CPU 使用单线程。CUDA 实测设备为 RTX 4090（sm_89）。

**结论：未更新 Yee 电场分量产生的伪物性梯度已修复并安装；公开 `compute` 对物理边界材料的完整导数仍为 UNRESOLVED。** 后者包含复制进入 PML 的材料链和重建 CPML 系数的链，不能通过冻结边界或屏蔽物理边缘宣称通过。

## 1. 实际调用链及离散伴随

`solver/modeling.py:compute` 先调用模型预处理，将物性复制到 PML，再增加 Yee halo，构造物性更新系数和 CPML 系数；最后在第 619 行调用 `autograd.DeepGPR.apply(config, tensors, eps_r, sigma, source, *states)`。`autograd.py:490` 的显式张量输入是扩展后的 eps/sigma、源波形及 30 个状态。CPML 系数位于 `SolverTensors` 中，作为普通 Python 对象参数传入。

CPU 正演每步顺序为：保存 Eold → H 基础更新 → H CPML → E 基础更新 → E CPML → 源注入 → RHS 历史/接收数据。反向循环先注入接收数据余切，再求源波形与物性余切，然后转置 E CPML/基础更新，最后转置 H CPML/基础更新。CUDA general 和 TM2D fast 路径执行相同时间顺序。物性核读取的 lambda 是完成接收注入后的当前 Enew 余切。

Yee curl 采用不同 forward/backward staggered 差分；靠近数组边界的降阶策略也要逐项转置。CPU 的 `staggered_*_transpose`、CUDA general 的散射/原子转置和 fast 路径的权重 gather 实现对应的离散转置，并非简单用连续算子的负 curl 代替。

单个 CPML 更新写成（D 已包含网格间距；s 为对应分量符号）：

```text
field_new = field_base + s * cb * ((RA - 1) * D + RB * phi_old)
phi_new   = RE * phi_old - RF * D

lambda_D       = s * cb * (RA - 1) * lambda_field - RF * lambda_phi_new
lambda_phi_old = s * cb * RB * lambda_field + RE * lambda_phi_new
```

CPU `deepgpr_cpu.c:512:pml_adjoint_weight` 精确实现后两式；CUDA general CPML 转置和 fast 权重更新也采用该链。这里的结论仅适用于**固定 RA/RB/RE/RF 和物性更新系数的状态转置**，不代表这些系数对物性的导数已经实现。

隔离的新库上，现有 `test_discrete_adjoint.py` CPU 7 项和 CUDA 9 项均通过，覆盖阶次 2/4/8、场更新、全脸/非对称 CPML、2D fast 状态点积、源波形 FD、内区物性 Taylor、重合源接收及不完整采样块。日志为 `discrete_native_cpu.log`、`discrete_native_cuda.log`。原 t02 状态验证的 44 个点积行均 PASS，状态 FD 为 20 PASS、1 INCONCLUSIVE。状态伴随通过与完整边界材料导数失败并不矛盾。

## 2. 物性系数链及已修复缺陷

对于实际执行基础 E 更新的分量：

```text
A  = epsilon0 * eps_r / dt + sigma / 2
ca = (epsilon0 * eps_r / dt - sigma / 2) / A
cb = 1 / A
Enew = ca * Eold + cb * R

dca/deps_r = epsilon0 * (1 - ca) * cb / dt
dcb/deps_r = -epsilon0 * cb^2 / dt
dca/dsigma = -(1 + ca) * cb / 2
dcb/dsigma = -cb^2 / 2
```

原代码对每个历史分量都累加 `grad_ca=lambda*Eold`，并假定 `R=(Enew-ca*Eold)/cb`，但实际 `update_e_cpu`/`update_e_gpu` 有分量特有的 Yee 更新条件。例如 Ez 的低 x/y 面、Ex 的低 y/z 面、Ey 的低 x/z 面可保留旧场而不执行 `ca*Eold`。

最小 CPU 复现：12×12、无 PML、一步，Ez[0,4]=1，零源，同位置接收，eps=4、sigma=0.01。接收数据始终为 1，J=0.5，与 sigma 无关。原 AD 为 **−0.7033979893**；独立正演 FD 在 h=0.003 和 0.001 均为 **0**。证据见 `unupdated_boundary_cpu_probe.json`。

正确的不更新分量映射是 `Enew=Eold+cb*Rcorrection`：其 `grad_ca` 应为零，而 cb 仍可通过边界源注入或 CPML correction 产生真实物性导数。直接跳过该分量会错误删除这些导数。

修复覆盖 CPU、CUDA ordinary（fast/general 共用，FP32/FP16/BF16）及 INT8 material-gradient 核：使用与基础 E 更新完全相同的分量条件，仅在执行更新时累加 ca 余切；reconstructed RHS 使用实际历史系数 ca 或 1。

另一个相关的存储缺陷也已修复：如果先将人工 `(1-ca)*Eold/cb` 存入 RHS，再在反向解码后减去它，R 与 E 的独立有损量化会残留伪导数。中间尝试在 INT8 零源例中留下 sigma≈−4.5054e−5；最终实现把正确 `R=(Enew-a*Eold)/cb`（a=ca 或 1）在存储/量化之前计算，普通、vec2、INT8 历史保存路径全部覆盖。Forward fields、源注入和状态转置未改变；仅历史 RHS 的不更新分量定义与对应余切改变。

关键修复位置：CPU `material_component_has_e_update_cpu`、`save_rhs_snapshot_cpu` 和 `accumulate_material_gradients_cpu`；CUDA `material_component_has_e_update_gpu`、`rhs_history_coefficient_gpu`、三个 RHS 保存/量化路径及两个物性梯度模板核。外部 ABI 未改变。

### 修复前后数值

| 合成一步例 | 原 sigma AD | 修复 sigma AD | 正演 FD | 解析值 |
|---|---:|---:|---:|---:|
| 无 PML、非零初始 Ez、零源，CPU/CUDA FP32 | −0.7033979893 | 0 | 0 | 0 |
| 同例，CUDA INT8 stored | −0.7034430504 | 0 | 0 | 0 |
| 未更新 Ey，保留 z0 CPML 辅助 correction=0.5 | −1.1179770231 | −0.1671946198 | −0.1672045570 | −0.1671946478 |

第三例 eps 修复 AD=−0.1184298024，FD=−0.1184453988，解析=−0.1184298250。它专门证明 cb correction 被保留。所选 CPML q=0 的 RA/RB/RE=1、RF=0，其 correction 不依赖 CPML 系数变化，因此可用于隔离此缺陷。测试也包含非零边界源 source=0.0002，避免只证明零梯度。

新增 `tests/test_unupdated_e_material_gradients.py`：CPU 3 个测试/22 子用例，CUDA 5 个测试/54 子用例，全部通过。包含 2D Ez、3D Ex/Ey/Ez、stored/reconstructed、非零源、CPML correction，以及 FP16/BF16 的 legacy/native_scalar/native_vec2 历史路径和 INT8 fast/general。日志和 before/after FD/AD 数据为 `native_fix_{cpu,cuda}_regression.log`、`native_fix_{cpu,cuda}_verification.json`。

### 正常内区合成 D 的旧新对照

使用零初态、原默认 D 模型：2D 48×48、nt=400、PML=10；3D 16×16×12、nt=150、PML=4。对 stored/reconstructed 两种历史、全部物理模型梯度进行对比；没有对梯度乘边界 mask。

| 路径 | 接收波形 | eps/sigma/source 梯度 |
|---|---|---|
| CPU 2D/3D | 全部逐位一致 | 全部逐位一致，差=0 |
| CUDA fast 2D | 逐位一致 | eps/source 逐位一致；sigma maxabs=1.49e−8、relative L2=4.69e−8 |
| CUDA general 2D | 逐位一致 | 最大 relative L2=3.47e−7 |
| CUDA general 3D | 逐位一致 | 最大 relative L2=6.23e−6（eps），sigma≤3.48e−7、source≤1.37e−6 |

CUDA general 采用已有原子累加路径，梯度逐位结果会随执行顺序变化；上表如实保留差异，没有将其归零。证据与完整张量分别存于 `normal_native_{cpu,cuda}_comparison.json` 和 `normal_native_{cpu,cuda}_arrays.pt`。这验证正常零初态正演不变，梯度变化处于很小的数值量级。

## 3. 完整 PML 材料导数：UNRESOLVED

三个独立断点仍存在：

1. `preprocessing/model_setup.py:72:_ExtendModel` 正演每次使用 replicate pad，反向第 90 行仅裁剪物理切片，不执行 replicate pad 的转置求和。边界物理像素变化会改变它复制到的 PML 材料，当前链丢失了这一贡献。
2. CPU/CUDA ordinary/CUDA INT8 物性梯度核显式排除所有 PML 位置。即便仅改 `_ExtendModel.backward` 为求和，它收到的 PML 物性梯度仍为零，因此无法补回完整复制链。
3. `solver/pml.py:245` 将 eps/mu detach，第 83 行 sigmamax 计算还有 no_grad；CPML 系数实际每次从当前边界平均值重建。原生 backward 只有 eps/sigma/source/state 余切接口，没有更新系数或 CPML 系数余切。`autograd.py:729` 对 `tensors` 参数返回 None。仅去掉 detach 也不会跨越自定义算子恢复此链。

`gv_common.py:420:edge_mask` 将带 CPML 面的最外层**物理**像素清零，`directions` 第 461 和第 481 行统一应用该 mask。因此大量内区方向通过只能证明固定边界的部分导数，不能证明公开 `compute` 的全模型梯度。用户明确禁止隐去这一边界因素，后续严格矩阵必须包含不屏蔽物理边界的方向。

读取原 `data/t04_pml.json` 时共 173 行：133 PASS、27 PASS(noise-limited)、9 FAIL、1 INCONCLUSIVE、3 无 verdict。9 个 FAIL 全为 `G_total_derivative`，来自 eps、sig、eps_edge_layer_only × CPU/CUDA fast/CUDA general。文件中的“expected to differ … by design”注释描述了部分导数设计，但不能满足本次公开 compute 完整导数要求。

CPU 原失败的代表值：

| 方向 | AD | 完整 compute FD | 差异 |
|---|---:|---:|---|
| eps random_unmasked | −0.0201238281 | −0.0130733482 | 约35% |
| sigma random_unmasked | −0.0100359872 | −0.0105981540 | 约5.3% |
| eps_edge_layer_only | −0.0028355201 | +0.0043047466 | 符号相反；Taylor slope≈1.136 |

本审计额外独立 CPU 复现 `reproduce_boundary_cpu.py`，G、48×48、nt400、PML10。h=0.01 时：

```text
AD                                     -0.00283552014
完整 compute FD                        +0.00430081759
固定 PML 材料与系数的 FD                 -0.00283602644
复制材料变化、固定 CPML 系数的 FD         +0.00430225358
缺失复制贡献                            +0.00713828002
```

数据 `boundary_cpu_reproduction.json` 包含 h=0.03/0.01/0.003/0.001。冻结边界实验仅用于定位丢失项，没有被用作完整梯度通过的判据。厚 PML10 的这个随机方向对系数重建敏感度很弱，不表示系数链可忽略：另一个 G 的均匀物理边缘方向诊断（h=0.03）在 PML2 上得到完整 FD=−0.1130963819，固定系数/变化复制材料 FD=−0.0817254611，系数重建贡献约 **−0.0313709208**。PML1 的贡献约−0.0064650658；探索数据见 `coefficient_cpu_probe.json`。这些单个步长的分解数值用于展示遗漏可显著，不单独充当收敛证明。

## 4. 完整修复的最小架构及成本

当前原生接口没有可直接折叠的 extended-updatecoeff gradient 输出。完整实现需要同时补上：

- 将物性历史/梯度覆盖到实际 PML 材料位置，保留当前物理-only history 模式的明确限制或引入扩展历史；再将复制 padding 的余切按实际 replicate 映射求和回边界，包括角/棱的多重复制。仅取消 PML 排除但不补历史不可行。
- 对每脸 CPML 平均材料建立可微链。当前只训练 eps/sigma 时，可以传入最多六个显式可微 eps 平均值，在原生累计六个标量 VJP；另一方案是返回每脸 RA/RB/RE/RF 余切，再由 Python 系数生成公式反传。两者都必须穿过 custom-autograd 的显式 tensor 输入/输出接口。
- 正演 CPML 系数余切需要当步 D 和 phi_old；当前 Eold/R 历史没有这些信息。不能在反向 phi 已被覆盖后推断它们，也不能只复用当前 adjoint phi。

例如把一脸平均值记作 m，sigma_max∝m^(−1/2)，所以 `d sigma_max/dm=−sigma_max/(2m)`。可在正演记录每个活跃 CPML 辅助更新的两个局部导数因子：

```text
F_field = s * cb * (dRA/dm * D + dRB/dm * phi_old)
F_phi   = dRE/dm * phi_old - dRF/dm * D
```

反向在转置/覆盖辅助余切前，将 `lambda_field*F_field + lambda_phi_new*F_phi` 累加到对应脸标量。这样无须存全部原始 D 和 phi_old，仍需额外保存两个 FP32 因子/辅助单元/时间步。若 Nphi 为全部电磁 CPML 辅助数组有效单元数，额外历史约 `8 * nt * Nphi` 字节（未计布局/对齐），系数导数数组和六个标量输出较小。直接系数 VJP 方案同样通常需两套 CPML 局部历史。

涉及 CPU CPML 正反向、CUDA general 电磁 CPML cells、CUDA fast 正演 fused CPML 和反向 weights、原生 ABI及 ctypes bridge、SolverTensors/DeepGPR 显式输入、历史分配器和 native libraries。电场、磁场、六个面、2D/3D 与 fast/general 都必须覆盖。无 mu 训练也不能省略 H CPML 系数对边界 eps 平均值的链。此改动明显超出本次安全最小 Yee 修复；本审计没有改变 forward 以冻结系数，没有只做 crop 折叠，也没有将完整 PML 标为通过。

## 5. 构建、安装与平台边界

旧库已完整备份到 `native_libraries/before/`。新库先编译到 `native_libraries/after/`，测试进程只覆盖 `loader.LIB_DIR`，并未改变工作区库。CPU/CUDA 验证完成后，按 root 授权通过同目录临时文件加 `os.replace` 原子安装到 `src/DeepGPR/lib/deepgpr_cpu.so` 与 `deepgpr.so`，安装 UTC 为 2026-10-10 14:15:17。

| 文件 | 最终 SHA256 |
|---|---|
| deepgpr_cpu.c | bae0dab09c1844042cc952fbafaba85de3417a90451282fa87e97ef89c1a2eca |
| deepgpr.cu | 848fdd85799767f8ad1712d566bd7b840d28848157b279d8ad3214e98b1ff34a |
| deepgpr_cpu.so | 2925ec1d49c36f4914f68442209e743d46ee34f764d3b4f5a7f986c0e3a39161 |
| deepgpr.so | d2e458bd438ed0a050ad7e74b871e5c8ad5ce6633e65954020e28dc998401861 |

构建命令与旧/新/已安装 hash 均保存在 `native_build_manifest.json`。CPU 为 Linux x86_64 OpenMP 构建；CUDA 使用 CUDA 12.4、`-arch=sm_89`。Windows/macOS 预编译库未重建；其他 GPU 架构未实测，其二进制适配不属于这里的通过结论。生产 C/CUDA 源包含修复，其他平台需要据此重新编译。外部 ABI6 保持兼容。

`git diff --check` 通过。当前 deepgpr 环境和系统 Python 均无 Ruff 模块，PATH 也没有 Ruff，因此未声称 Ruff 已执行。这里未触碰 TV/网络/Checkpoint 模块，相关报告由对应审计提供。
