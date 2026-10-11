# DeepGPR 全链路梯度正确性审计（2026-10-10）

**结论：当前实现不能被认定为 public `compute()` 的完整物性导数正确。内部物性方向、固定材料下的场状态伴随与 Checkpoint 等价性有通过证据；包含物理边界的物性扰动存在明确失败。两处独立缺陷已修复，PML 材料复制与系数的总导数仍未修复。**

本次全部输入为程序生成的合成模型/波形，没有读取或执行真实 GPR 数据、桌面实验代码或任何既有反演实验。原有未提交测试及日志被保留，所有新证据位于本目录。当前分支为 `test/gradient-verification`，起点 `3bfc1e998851dbe489cf3a56b3a7cb27d8f476ba`。主体测试于 10 月 10 日执行，冻结物性补证及报告核对于 10 月 11 日完成；本次修改尚未提交或推送。

## 1. 实际代码结构与梯度传播

| 模块 | 当前实现与梯度角色 |
|---|---|
| 公共入口 | `src/DeepGPR/solver/modeling.py::compute`，验证物性/几何，选择 CPU、CUDA general 或二维 TM fast |
| 物性扩展 | `preprocessing/model_setup.py::initialization` / `_ExtendModel`，物理模型 → 复制到外部 PML → Yee halo；当前 backward 只 crop |
| 材料系数 | `lib/deepgpr_cpu.c`、`lib/deepgpr.cu`，按同一材料索引计算各 E 分量的衰减/旋度系数，没有跨 Yee 物性平均；εr 为相对量，链式法则包含 ε0 |
| CPML | `solver/pml.py::build_pml_coeffs` / `calculate_pml_update_coeffs`，6 面 E/H 系数与 24 个辅助状态；边界均值 detach |
| 自定义反传 | `solver/autograd.py::DeepGPR` 保存 E / R 历史、材料及 CPML 系数，native backward 返回物性、源、30 个初始状态的 cotangent |
| Native 调用 | `native/abi.py`、`bridge.py`、`loader.py`，ctypes ABI 与 CPU/CUDA 动态库 |
| 内存/精度 | `solver/storage.py`、`memory.py`、`sampling.py`；传播始终 fp32，历史支持 fp32/fp16/bf16、CUDA INT8、异步 pinned CPU offload |
| 时间 Checkpoint | examples 和测试按时间分段传递全部 30 状态，Torch reentrant / nonreentrant 重算；in-place native 状态必须 clone |
| Loss / 正则 | Torch L1/MSE；`postprocessing/filters.py` 的 FIR/Hilbert；`inversion/regularization.py::TVRegularization` |
| 网络 | 当前 src/examples 没有 Single/Dual-SIREN、M-SSIM、OT、互相关损失实现；独立测试 fixture 不代表产品已有这些功能 |

梯度链：网格/网络参数 → εr、σ → float32 cast → 物性扩展 → Yee/CPML native 正演 → 接收器数据 → float64 目标/实际例子 Loss → 反传 receiver cotangent → 反序 receiver/source/E-CPML/E-update/H-CPML/H-update → native 材料与初态梯度 → crop → 参数。源与接收器按整数 Yee 索引 scatter/gather；没有要测试的亚网格插值权重。重复位置的累加和转置由现有回归测试单独检查。

局部电场系数可写为 `a=σ dt/(2 ε0 εr)`、`ca=(1-a)/(1+a)`、`cb=dt/(ε0 εr(1+a))`；场采用 Yee 交错差分，各轴 spacing 独立。验证对象是实际逐点材料索引实现，不用连续伴随公式替代离散梯度检查。

## 2. 环境、配置与独立性

- **python**：3.10.16
- **torch**：1.11.0+cu115
- **torch_cuda**：11.5
- **gpu**：NVIDIA GeForce RTX 4090
- **gpu_capability**：[8, 9]
- **nvidia_smi**：NVIDIA GeForce RTX 4090, 575.51.03, 68 %, 1186 MiB, 24564 MiB
- **solver_precision**：float32 (native CPU and CUDA solvers have no float64 path)
- **misfit_precision**：float64 accumulation of float32 receiver data
- **seed**：20261010
- **omp_num_threads**：1

模型 A–G：均匀；ε 异常；σ 异常；双异常；水平层；倾斜界面；贴近边界异常。二维 40×40，三维 14×14×10；dx=0.02 m；默认二维 dt=25 ps、三维 dt=20 ps。εr 最低 4、σ 最低 0.005–0.01 S/m，所选 dt 低于对应 CFL 上界；源/接收器在物理网格内，确定性 Ricker 400 MHz、2 个炮次；观测由不同物性模型独立正演生成，未用同模型零残差代替验证。

FD 使用 `torch.no_grad()`、禁用历史的独立 native 正演目标，绝不调用被测 backward。公共 `compute()` 每个扰动都正常重建 PML，没有冻结 PML 或遮掉边界来让主测试通过。`interior_local` 是额外的紧支撑诊断，和完整随机/平滑/边界方向并列保存。物性方向乘相应 ε/σ 均值做尺度转换，h=0.3 到 1e-4；只去掉违反 εr≥1、σ≥0 的步长。

主 physical 矩阵的预设相对容差 2e-3，要求相邻步长或可解析的外推一致，并检查 Taylor 余项斜率与测得的 fp32 噪声。只有误差最小的单点不构成 PASS。接近零、有效余项不足或有限差分受量化限制时保留 INCONCLUSIVE。所有 h、J±、绝对误差、相对误差、噪声、梯度范数与判定原因都在 JSON；CSV 是摘要，不能替代原始曲线。

Checkpoint/GPU 存储专项另有初始脚本中固定的辅助 FD 判据：best<0.5%，且至少两个 h<1%。它比主矩阵宽，表中辅助 PASS 只代表该规则，不算 0.2% 严格认证，也不纳入主 physical 通过计数；无 PML 弱 σ 方向另作分辨率复核。Network FD 使用相邻稳定/可分辨的 0.2% 判据；loss 的 float64 gradcheck 参数与每项独立差分容差都保存于 loss JSON。判据按专项分别记录，未在修复后放宽。

## 3. 已执行测试矩阵

| 证据 | 执行范围 | 状态计数 |
|---|---|---|
| basic_after_native_fix.json（修复前另存 basic.json） | 二维 A–G，CPU / CUDA fast / CUDA general，ε/σ 像素、随机、局部、平滑；joint 与边界方向 | {'PASS': 87, 'FAIL': 105, 'INCONCLUSIVE': 45} |
| three_d.json | 三维 A–G，CPU / CUDA general，同类方向 | {'INCONCLUSIVE': 70, 'PASS': 28, 'FAIL': 46} |
| time_cpu.json | nt / 固定传播时间窗 / 固定总时长 dt，或 PML 厚度、非对称面、高阶差分 | {'INCONCLUSIVE': 10, 'PASS': 34, 'FAIL': 3} |
| time_cuda.json | nt / 固定传播时间窗 / 固定总时长 dt，或 PML 厚度、非对称面、高阶差分 | {'INCONCLUSIVE': 10, 'PASS': 34, 'FAIL': 3} |
| pml_cpu.json | nt / 固定传播时间窗 / 固定总时长 dt，或 PML 厚度、非对称面、高阶差分 | {'PASS': 25, 'INCONCLUSIVE': 8, 'FAIL': 16} |
| pml_cuda.json | nt / 固定传播时间窗 / 固定总时长 dt，或 PML 厚度、非对称面、高阶差分 | {'PASS': 26, 'INCONCLUSIVE': 7, 'FAIL': 16} |
| checkpoint_results_after_native_fix.json（原始结果另存） | 三维全部 30 非零初态；K=1/2/3/5 × plain/reentrant/nonreentrant 等；计数仅 tests 数组 | {'PASS': 17, 'MEASURED_APPROXIMATION': 6}；物性 FD 另为 {'FAIL': 3, 'PASS': 1}，state dot 另存 |
| loss_results.json | float64 满 Jacobian gradcheck + 3 方向多 h，随机/Ricker/平移/缩放/低幅；NumPy 独立前向参考 | {'PASS': 114, 'FAIL': 0, 'INCONCLUSIVE': 13} |
| gpu_storage.json | CUDA 精度/offload/checkpoint，详见 [Checkpoint 审计](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/checkpoint_audit.md) 与 JSON | 已执行项逐一记录，近似不等于正确性保证 |
| network_results.json | 36 个 Grid FWI / Single/Dual fixture 配置，92 参数方向 × data/TV/total；分支、α、映射与 activation checkpoint | {'PASS': 139, 'FAIL': 64, 'INCONCLUSIVE': 73} |
| grid_fwi_freeze.json | 12 个真实 ε-only / σ-only / joint Grid FWI 配置，20 方向 × data/TV/total；冻结另一物性并只更新 active 参数 | {'PASS': 28, 'FAIL': 14, 'INCONCLUSIVE': 18} |

物理主矩阵共有 573 个方向测试：{'PASS': 234, 'FAIL': 189, 'INCONCLUSIVE': 150}。该计数不混入修复前重复、采样像素、loss、网络或存储等价检查，也不把多个 h 当作独立通过项。

## 4. 方向导数数值汇总

下表有意同时列内部通过与完整边界失败。FD 栏为扫描中最佳解析点；状态还取决于相邻步长、Taylor 与噪声，完整扫描不可省略。

| 测试项目/方向 | 参数 | AD | FD | 相对误差 | 状态 |
|---|---|---:|---:|---:|---|
| 2d_D / local (cuda_fast) | eps | -0.43684 | -0.436847 | 1.66082e-05 | PASS |
| 2d_D / local (cuda_fast) | sig | -0.0200819 | -0.0200824 | 2.31273e-05 | PASS |
| 2d_D / interior_local (cuda_fast) | joint | -0.456927 | -0.456921 | 1.33425e-05 | PASS |
| 2d_D / smooth (cuda_fast) | eps | -0.199856 | -0.268628 | 0.256012 | FAIL |
| 2d_G_boundary / edge (cuda_fast) | eps | 0.030698 | -0.421096 | 1.0729 | FAIL |
| 2d_G_boundary / edge (cuda_fast) | sig | -0.00832204 | -0.0149544 | 0.443508 | FAIL |
| 3d_D / local (cuda_general) | eps | -0.00473827 | -0.00474022 | 0.000412297 | PASS |
| 3d_D / local (cuda_general) | sig | -0.000211627 | -0.000211351 | 0.00130556 | INCONCLUSIVE |
| 3d_D / interior_local (cuda_general) | joint | -0.00461594 | -0.00461884 | 0.000626621 | PASS |
| 3d_G_boundary / edge (cuda_general) | eps | 0.00887789 | 0.0282914 | 0.686198 | FAIL |
| Checkpoint 物性参考 / entire_physical_model | eps | -0.0529332 | -0.0601038 | 0.119303 | FAIL（辅助判据） |
| Checkpoint 物性参考 / outer_physical_faces | eps | 0.0028382 | 0.00382162 | 0.25733 | FAIL（辅助判据） |
| Checkpoint 物性参考 / entire_physical_model | sig | -0.150179 | -0.149847 | 0.0022105 | PASS（辅助判据） |
| Checkpoint 物性参考 / outer_physical_faces | sig | -0.000311281 | -0.000624045 | 0.501189 | FAIL（辅助判据） |
| L2 (official torch MSELoss) (random) | 波形/正则输入 | 0.00227329 | 0.00227329 | 5.49425e-14 | PASS |
| L1 (official torch L1Loss) (random) | 波形/正则输入 | -0.006316 | -0.006316 | 2.10111e-14 | PASS |
| Envelope p=1 followed by MSE (random) | 波形/正则输入 | 0.00665815 | 0.00665815 | 1.34821e-11 | PASS |
| FIR cutoff=0.12 followed by MSE (random) | 波形/正则输入 | 5.31484e-06 | 5.31484e-06 | 8.1394e-12 | PASS |
| TV default-dtype contract (anisotropic) | 波形/正则输入 | -5.74624 | -5.74624 | 1.09742e-14 | PASS |
| M-SSIM / OT / 互相关 | 波形 | — | — | — | N/A：当前实现不存在 |
| 当前产品 Single-SIREN / Dual-SIREN | 权重 | — | — | — | N/A：当前实现不存在；fixture 结果单列 |
| Grid-FWI, pml=0 / joint, data+TV | 参数 | 2.50857e+06 | 2.50863e+06 | 2.1596e-05 | PASS |
| SingleSiren fixture, pml=0 / all, data+TV | 参数 | 1.6594e+06 | 1.65936e+06 | 2.68848e-05 | PASS |
| DualSiren fixture, pml=0 / low, data+TV | 参数 | -733608 | -734292 | 0.000931271 | PASS |
| DualSiren fixture, pml=0 / high, data+TV | 参数 | 619205 | 619552 | 0.000559298 | PASS |
| DualSiren fixture, pml=0 / alpha, data+TV | 参数 | 296504 | 296460 | 0.000147649 | PASS |
| DualSiren fixture, pml=0 / all, data+TV | 参数 | -572477 | -572498 | 3.60589e-05 | PASS |
| Grid-FWI, pml=6 / joint, data+TV | 参数 | -243714 | -244974 | 0.00514529 | FAIL |
| SingleSiren fixture, pml=6 / all, data+TV | 参数 | -165771 | -163544 | 0.0134343 | INCONCLUSIVE |
| DualSiren fixture, pml=6 / low, data+TV | 参数 | 654079 | 644985 | 0.0139035 | FAIL |
| DualSiren fixture, pml=6 / high, data+TV | 参数 | 230058 | 229498 | 0.00243138 | FAIL |
| DualSiren fixture, pml=6 / alpha, data+TV | 参数 | -69130.1 | -71896 | 0.0384705 | FAIL |
| DualSiren fixture, pml=6 / all, data+TV | 参数 | 195561 | 190368 | 0.0265561 | FAIL |
| Grid FWI 真正 epsilon_only, pml=0, MSEx1e4+TV | epsilon | -1.39255e+06 | -1.39257e+06 | 1.31383e-05 | PASS |
| Grid FWI 真正 sigma_only, pml=0, MSEx1e4+TV | sigma | 23962.6 | 23960.4 | 9.30252e-05 | PASS |
| Grid FWI 真正 joint, pml=0, MSEx1e4+TV | joint | 911301 | 911291 | 1.03667e-05 | PASS |
| Grid FWI 真正 epsilon_only, pml=6, L1x1e7+TV | epsilon | -6.62302e+06 | -6.55007e+06 | 0.0110151 | FAIL |
| Grid FWI 真正 sigma_only, pml=6, MSEx1e4+TV | sigma | -4022.03 | -3801.62 | 0.0548002 | FAIL |
| Grid FWI 真正 joint, pml=6, MSEx1e4+TV | joint | -249048 | -233878 | 0.0609099 | FAIL |

### 梯度范数与余弦

CPU/CUDA 整格 AD 一致性余弦在 basic.json 的 solver_comparison 中；这检验后端一致，不独立证明物性导数。真正 FD 像素向量采用 21 个预先选定的内部/边界像素，属于采样梯度余弦，不能称全网格 FD 余弦。

| 采样参数 | 区域 | 样本数 | AD 范数 | FD 范数 | 余弦 | 相对范数误差 |
|---|---|---:|---:|---:|---:|---:|
| eps | all | 21 | 0.00724312 | 0.00733684 | 0.985643 | 0.168851 |
| eps | interior | 16 | 0.00722794 | 0.00722769 | 1 | 7.5725e-05 |
| sig | all | 21 | 0.144257 | 0.15483 | 0.969125 | 0.249391 |
| sig | interior | 16 | 0.1389 | 0.138899 | 1 | 0.000124207 |

## 5. 时间长度、PML、Checkpoint 与存储精度

nt=50/100/200/400/800 分别生成合成观测，短窗信号不足时标 INCONCLUSIVE。损失窗口测试始终传播 800 步，仅改变 [start,end)；dt 测试保持总时长 10 ns，(dt,nt)=(50 ps,200)/(25 ps,400)/(12.5 ps,800)。Checkpoint K 单独改变，不与 nt/window/dt 混合。内部误差没有显示一致的随 nt 恶化趋势；完整平滑方向在波到达 PML 后明显失配，不能据此归因成普通时间累积误差。

固定材料和固定 CPML 的 30 状态线性算子点积是物性 FD 的补充：CPU 3D nt=1/5 点积误差分别 3.727e-7 / 8.015e-8；现有 CUDA 多阶/多面状态转置、源/接收器重复位置等回归另有日志。点积通过不包含 d(PML材料/系数)/dm，所以不消除边界物性 FAIL。

CPU 12 个分段组合中波形与全部终态逐位相同，物性/源/初态梯度差只在 fp32 累加精度内。去掉 clone 的负对照明显出错，说明原地状态不能直接复用。异步 offload 只在 CUDA 有效，CPU 请求为 no-op；INT8 为 CUDA-only，异步+INT8 不支持；ZFP 当前没有实现，N/A。

Checkpoint 专项的无 PML 弱 σ 边界方向仍保留原始 FAIL。随后 200 个独立前向探针复现原值：重复运行噪声为零，但 ±1 ULP 材料扰动引起的目标跳变与 FD 分子误差相当，相邻 h 的误差变号。因此派生审阅为 INCONCLUSIVE_FLOAT32_RESOLUTION，不能计作 PASS；实测 ULP 指标不是严格误差界。原阈值和 JSON 未修改，详见 checkpoint_fd_resolution_review.json / checkpoint_audit.md。PML 总导数失败不受此复核影响。

低精度保存改变反向历史，正演仍 fp32。fp16/bf16/INT8 的梯度差按真实全格误差、余弦与 FD 逐个记录，近似量化没有通用 PASS 保证。小模型的峰值显存/时间不是大模型性能基准。下表峰值仅为 PyTorch allocator，**不包含 native cudaMalloc 或驱动占用**，不能当作进程完整 GPU 显存。各后端 stored/auto/E-only、sampling interval 与异步结果参见 GPU JSON 和专项报告。

| 几何 | 配置 | ε 全格梯度差 | σ 全格梯度差 | 波形逐位相同 | 峰值 allocated MiB | 总时间 s | 存储判定 |
|---|---|---:|---:|---|---:|---:|---|
| 2D | stored_fp32_baseline | 0 | 0 | True | 17.4678 | 0.0143591 | PASS |
| 2D | fp32_auto | 0 | 0 | True | 9.23682 | 0.0148748 | PASS |
| 2D | fp16_legacy_stored | 2.84688e-05 | 3.38468e-05 | True | 9.21582 | 0.0147784 | MEASURED_APPROXIMATION |
| 2D | bf16_legacy_stored | 0.000242934 | 0.000276117 | True | 9.21582 | 0.0150465 | MEASURED_APPROXIMATION |
| 2D | async_fp32_auto | 0 | 0 | True | 0.984863 | 0.0159415 | PASS |
| 2D | int8_stored | 0.000637767 | 0.000792878 | True | 5.38965 | 0.0152402 | MEASURED_APPROXIMATION |
| 2D | sampling_3 | 2.448e-06 | 4.19143e-06 | True | 6.49316 | 0.0132459 | MEASURED_APPROXIMATION |
| 2D | sampling_5 | 3.59146e-06 | 4.94126e-06 | True | 4.26465 | 0.0127248 | MEASURED_APPROXIMATION |
| 3D | stored_fp32_baseline | 0 | 0 | True | 61.9648 | 0.0116322 | PASS |
| 3D | fp32_auto | 4.3488e-07 | 1.30246e-07 | True | 32.1113 | 0.0112553 | PASS |
| 3D | fp16_legacy_stored | 5.81266e-05 | 2.15475e-05 | True | 31.2617 | 0.0141913 | MEASURED_APPROXIMATION |
| 3D | bf16_legacy_stored | 0.000791145 | 0.000196941 | True | 31.2617 | 0.0130896 | MEASURED_APPROXIMATION |
| 3D | async_fp32_auto | 3.64775e-07 | 1.75085e-07 | True | 2.11133 | 0.0145271 | PASS |
| 3D | int8_stored | 0.0196134 | 0.00747269 | True | 17.5293 | 0.0151907 | MEASURED_APPROXIMATION |
| 3D | sampling_3 | 0.00111444 | 9.01298e-05 | True | 21.9648 | 0.0115871 | MEASURED_APPROXIMATION |
| 3D | sampling_5 | 0.00123512 | 0.000151384 | True | 13.6836 | 0.0118193 | MEASURED_APPROXIMATION |

GPU 78 行配置/Checkpoint 全部执行成功，22 行精确存储/重算等价 PASS，56 行为 MEASURED_APPROXIMATION，另 6 项不支持组合的拒绝检查通过。3D 的 fp16 最坏全格误差约 0.01686%、bf16 约 0.07914%；3D INT8 stored ε 差约 1.96%、σ 差约 0.75%。这些是相对 FP32 的全格梯度 L2 差，与方向 FD 的 0.2% 容差属于不同指标，不能直接据此判定方向导数 PASS/FAIL。独立 compact/outer-face FD 扫描另存于每项配置，严格基准使用 FP32、S=1。完整进程 CUDA 峰值未测定，该性能项目仅有 allocator 部分测量，不能宣称已验证总显存峰值。

## 6. Loss 与 FWI / IFWI 链路

[损失专项报告](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/loss_audit.md) 包含完整 127-case 表与修复前矩阵。MSE、L1、Envelope/FIR 组合的平滑点通过；精确 L1 零残差、p=1 envelope 的零幅、无平滑 TV 的零差分是不可微点，标 INCONCLUSIVE。另 8 个极低幅 TV 的双精度差分无法从常数底与消减噪声中可靠分辨，也保留 INCONCLUSIVE。

实际 2D 例子：L1×1e7 + TV；3D 例子：MSE×1e4 + TV。独立 loss 通过不代表 FDTD 组合通过；network_results 的 Grid FWI 为 joint 配置中的 ε、σ 及联合扰动方向，Single/Dual fixture 另含冻结输出物性的模式。[网络专项报告](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/network_audit.md) 明确区分 repo 实现与新增测试用 SIREN fixture，包含每个 branch、α、映射、checkpoint 与 NaN/Inf/None 检查。任何 fixture 的 PML 失败都是链路真实结果，不能用 loss 下降或图片替代。

另补 [真实冻结物性的 Grid FWI 专项](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/grid_fwi_freeze_audit.md)：ε-only / σ-only / joint × PML=0/6 × L1/MSE，共 12 个配置。前两种模式的另一物性 requires_grad=False，active 参数列表明确排除它。12 项完整 combined backward、active 有限梯度、分项和一致性、真实 clone-only SGD 及原物性保持不变均通过；8 项单物性模式的 inactive 梯度为 None 且全程逐位固定，这是预期冻结行为。它们的 60 个 FD 检查独立保存，启用 PML 后仍出现失败。旧 network 36-case 的脚本/数据/日志/报告起止 hash 未变。

36 个配置均以新鲜 FDTD 图执行一次实际 combined data+TV backward，逐参数对比独立 data-gradient+TV-gradient：{'cases': 36, 'all_parameter_comparisons_pass': 36, 'identical_forward_values': 36, 'max_gradient_difference_norm': 1.3332629595737868e-07}。随后对独立参数副本执行真实 SGD.step，并核对更新公式和原参数未改：{'cases': 36, 'matching_updates_and_unchanged_originals': 36, 'cases_with_actual_parameter_change': 36, 'loss_descent_evaluated': False}。这仅证明优化器使用了测得的梯度；不修复或证明 PML 导数，也没有用 loss 下降作判据。

| 链路 | PML | data/TV/total 方向检查计数 |
|---|---:|---|
| Grid-FWI | 0 | {'INCONCLUSIVE': 8, 'PASS': 10} |
| Grid-FWI | 6 | {'FAIL': 10, 'PASS': 5, 'INCONCLUSIVE': 3} |
| SingleSiren | 0 | {'PASS': 19, 'INCONCLUSIVE': 5} |
| SingleSiren | 6 | {'FAIL': 10, 'PASS': 7, 'INCONCLUSIVE': 7} |
| DualSiren | 0 | {'PASS': 69, 'INCONCLUSIVE': 27} |
| DualSiren | 6 | {'FAIL': 44, 'PASS': 29, 'INCONCLUSIVE': 23} |

网络输出物性映射 float64 Jacobian gradcheck 2/2 通过；activation checkpoint 8/8 目标与梯度逐位一致。无 PML 权重检查没有确认的 FAIL，但仍有 INCONCLUSIVE；启用 PML 后 low/high/α 等出现明确 FAIL。网络传播的原生 FDTD 仍是 float32，不能把网络 float64 参数误称双精度波场验证。

## 7. 实际缺陷、修复及未解决项

### 已修复：TV 输出 dtype

`regularization.py` 原先用默认 dtype 的 torch.tensor(0.0) 累加，float64 输入却得到 float32 目标，AD 没有对应前向量化导数。修复为 `reference.new_zeros(())`。3 类默认 dtype TV 的严格 gradcheck/FD 从 FAIL 变为 PASS；12 个 float32 fixture 的目标和双参数梯度逐位不变；12 项 loss 回归通过。

### 已修复：未更新电场分量的虚假 σ 梯度

边界 Yee 条件没有执行基底 E 更新时，E 应保持 Eold，再可能接受 cb-scaled source/CPML correction。旧材料核仍用 Enew=ca Eold+cb R 解释所有分量。无 PML、12×12、1 步，Ez[0,4]=1、同点接收的 J 对 σ 恒定，旧 AD=-0.70339799、FD=0。修复按实际分量更新谓词：未更新时 grad_ca=0；Rcorrection=(Enew−Eold)/cb，保留源和 CPML 对 cb 的依赖。CPU、CUDA 普通/低精度/INT8 物性核同步修改，不改 forward 更新。修复前后复现、独立库构建和回归见 [Native 审计](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/native_audit.md)。

### UNRESOLVED：公共 PML 总导数遗漏

存在两部分：① native 不累计 CPML 材料梯度，_ExtendModel crop 丢失复制材料到物理边界的 transpose；② build_pml_coeffs.detach/no_grad 及 native backward 接口丢失 CPML 系数导数。每个新的正演却会随边界 ε 重建这些值。文档已经说明 fixed-PML 偏导语义，但按用户要求检验公共正演的完整导数，仍是 FAIL。不能用文档约定把数学不相等改称 PASS。

CPU 最小复现中 edge-only AD=-0.00283552，h=0.01 的完整 FD=+0.00430082，方向反号；同一步长固定外部材料/系数 FD=-0.00283603，仅复制材料且固定系数 FD=+0.00430225。冻结只用于分解原因，没有替换主测试目标。另一 pml=2 方向：AD=+0.04127896，完整 FD=-0.11309638；固定系数但仍复制材料的 FD=-0.08172546，遗漏的系数贡献约 -0.03137092。两个缺失项都实际显著。

完整修复需要材料扩展的真正 transpose、CPML coefficient VJP 与 forward φold/旋度历史或精确 replay，覆盖 CPU gather、CUDA general scatter、TM fast、压缩/卸载和 ABI。本次没有实施一个仅折叠 crop、冻结新 forward 或用数值 backward 的不完整补丁。该项保持 UNRESOLVED；后续修复必须保持原正演并再次执行全部 public-FD 矩阵。

## 8. 图表

### FD 步长与 Taylor 余项

![FD 步长与 Taylor 余项](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/fd_convergence.png)

### 传播 nt 与损失窗

![传播 nt 与损失窗](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/time_and_windows.png)

### 时间窗余弦与固定总时长 dt

![时间窗余弦与固定总时长 dt](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/window_cosines_and_dt.png)

### Checkpoint 对照

![Checkpoint 对照](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/checkpoint_comparison.png)

### εr/σ 采样 AD–FD 梯度散点

![εr/σ 采样 AD–FD 梯度散点](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/gradient_scatter.png)

### Loss 梯度检查

![Loss 梯度检查](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/loss_gradients.png)

### PML 厚度与边界方向

![PML 厚度与边界方向](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/pml_boundary.png)

### GPU 历史精度与 allocator 显存对照

![GPU 历史精度与 allocator 显存对照](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/figures/gpu_storage_accuracy_memory.png)

图中最佳扫描误差仅供观察；PASS/FAIL/INCONCLUSIVE 以原始 JSON 的多步判定为准。对数图中的精确零值以绘图下限显示，原 JSON 保留真实零值。所有图同时保存 PNG 与 SVG，可单独导出。

## 9. 最终十项回答

1. **2D/3D FDTD 梯度是否正确？** 固定物性/CPML 下的状态离散伴随有通过证据，内部物性方向通过；公共边界物性总导数 FAIL，不能整体宣称正确。
2. **εr 与 σ 是否分别通过？** 二维两者均有内部通过与边界失败。三维 A–G 主矩阵中 ε 与 compact joint 有通过证据，纯 σ 为 42 INCONCLUSIVE / 16 FAIL / 0 PASS；GPU 存储专项的另一紧支撑 3D σ 方向有通过证据，但不能代替整个主矩阵或普遍认证三维 σ。未更新场的伪 σ 梯度已单独修复。
3. **双参数联合梯度？** 内部联合方向可通过；完整随机/平滑联合方向可失败。
4. **是否随传播时间增加误差？** 所测内部方向没有统一增长趋势；早期小信号多为 INCONCLUSIVE，边界影响在后期显现。不能外推无限传播时长。
5. **PML 是否影响正确性？** 是。辅助状态 transpose 与物性/系数总导数必须区分，后者尚未解决。
6. **Checkpoint 是否改变梯度？** 完整传递并 clone 30 状态的已测分段只引入 fp32 累加差异，保留原梯度；原梯度的 PML 缺失也会保留。
7. **float16/bfloat16 是否不可接受？** 所测 3D 相对 FP32 全格误差最坏 fp16≈0.01686%、bf16≈0.07914%，没有显示很大的量化偏差，但不能普遍保证；INT8 stored ε≈1.96%、σ≈0.75% 明显更大。全格误差不直接等于方向 FD 判据，近似结果保留 MEASURED_APPROXIMATION；严格 FD 基准使用 fp32、S=1。
8. **各 Loss 自动梯度？** 所实现损失与信号变换的平滑案例通过，TV dtype 缺陷修复；不可微/极低幅分辨率不足案例仍 INCONCLUSIVE。未实现 loss 为 N/A。
9. **Single/Dual-SIREN？** 当前产品实现不存在，N/A；新增合成 fixture 的权重/分支/α/checkpoint 链路结果单列，不能冒充已有功能验证。
10. **通过和剩余问题？** Loss 修复后平滑梯度、状态伴随、正确状态保存的分段等有证据；边界 public-PML 总导数 UNRESOLVED，低精度/小梯度仅在各已测条件内评估。

## 10. 修复后完整回归与交付范围

完整 unittest discover：CPU {'status': 'PASS', 'tests': 151, 'executed': 113, 'skipped': 38, 'seconds': 10.591}；可访问 GPU 的 CPU/CUDA {'status': 'PASS', 'tests': 151, 'executed': 150, 'skipped': 1, 'seconds': 13.244}。两个日志分别保存，跳过计数保留；默认 unittest 日志不包含各跳过项的理由，支持性限制另见测试源码和专项报告。新增未更新场边界回归含 76 个子用例；现有离散伴随专项 CPU 7 / CUDA 9 项通过。精确命令与当前源文件/二进制 hash 见 final_verification.json。

Ruff 对 src/DeepGPR、tools 及两个修改/新增回归文件通过。正常 D 模型二维/三维修复前后波形逐位一致，CPU 梯度逐位一致；CUDA 最大相对梯度差 6.23e-6，和原子累加的数值差异相符。前后数组、旧库与新库 hash 均保留。未测试 Windows/macOS 二进制、其他 GPU 架构、大规模模型或真实反演；整体 CUDA 内存峰值未测定。

## 11. 可复现文件与命令

`physical_verification.py` 使用原有 gv_common 的纯合成模型/获取布局与正演包装，自己创建不遮罩的方向；不导入原 t06 或外部实验。`loss_verification.py`、`checkpoint_verification.py`、`gpu_storage_verification.py`、`network_verification.py` 以及 native regression 各自保存精确命令、环境与种子。

```bash
cd /home/ll/下载/DeepGPR-dev
PY=/home/ll/anaconda3/envs/deepgpr/bin/python
AUDIT=tests/gradient_verification/codex_audit_20261010
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py basic --tag after_native_fix
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py time --paths cpu --tag cpu
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py time --paths cuda_fast --tag cuda
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py pml --paths cpu --tag cpu
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py pml --paths cuda_fast --tag cuda
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py 3d --paths cpu,cuda_general
OMP_NUM_THREADS=1 $PY -u $AUDIT/physical_verification.py scatter --paths cuda_fast
OMP_NUM_THREADS=1 $PY -u $AUDIT/loss_verification.py
OMP_NUM_THREADS=1 $PY -u $AUDIT/checkpoint_verification.py --tag after_native_fix
OMP_NUM_THREADS=1 $PY -u $AUDIT/gpu_storage_verification.py
OMP_NUM_THREADS=1 $PY -u $AUDIT/network_verification.py
OMP_NUM_THREADS=1 $PY -u $AUDIT/grid_fwi_freeze_verification.py
OMP_NUM_THREADS=1 $PY -u $AUDIT/checkpoint_fd_resolution_review.py
OMP_NUM_THREADS=1 PYTHONPATH=src $PY -m unittest discover -s tests -p 'test_*.py'
$PY $AUDIT/build_report.py
```

GPU 命令需可访问宿主 NVIDIA 驱动；本次默认沙盒不可见驱动，实际 GPU 运行使用已授权的宿主执行。没有终止其它进程。运行前后原始动态库 hash/修复前备份与独立编译产物均保留，正演不变验证见 native 报告。Linux CPU/CUDA 库已经更新；CUDA 库用 CUDA 12.4、sm_89 构建，运行环境 Torch 1.11/cu115 实测成功。Windows/macOS 库及其他 GPU 架构没有重建验证，不能算通过。重跑本目录会更新本审计输出，原 tests/gradient_verification/data 不会被改写。

原始证据：[全物理摘要 CSV](/home/ll/下载/DeepGPR-dev/tests/gradient_verification/codex_audit_20261010/data/physical_summary.csv)、data/*.json、logs/*.log。此前原有日志中的 masking、外部 MSSIM 实验和宽容的量化 PASS 没有被纳入本次通过计数。未执行的后端/操作系统/真实反演不在结论范围内。
