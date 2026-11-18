# SpinQuant Jittor 本轮实验结果

本报告只汇总 **A + W16A4KV4，seed=0，2026-11-18** 这一次完整实验。历史实验及已终止的 B 方案不合并计算。PPL 越低越好；报告保留 6 位小数，CSV/JSON 保留原始精度。

## 完成情况与结论

- 正式任务 32/32 完成：4 次全新的 100 步旋转训练、28 项 PPL，形成 14 组官方 PyTorch / Jittor 对照。
- 每次训练完整记录 100 条逐步 Loss，总计 400 条。无失败、跳过或缺失，全部正式进程退出码为 0。
- SpinQuant 后的 6 组 PPL 对照绝对相对差异均小于 0.43%；两个浮点基线的差异均小于 0.001%。这支持本轮条件下最终 PPL 较好对齐。
- 直接 W4A4KV4 量化仍有明显差异：TinyLlama RTN 为 −2.5203%，TinyLlama GPTQ 为 +5.3664%，Mistral RTN 为 +5.4522%。不能写成全部配置严格一致。
- 训练逐步 Loss 形态接近，但初始固定探针和训练后固定探针均非逐位一致；Mistral Jittor 的固定探针 Loss 还略有上升。不能只靠最终 PPL 推断梯度、旋转或训练过程完全相同。


开始时间：2026-11-18T05:15:44.353803+08:00；结束时间：2026-11-18T11:33:16.273685+08:00（北京时间）。总历时 **6.2922 小时，约 6 小时 17 分 32 秒**，含预检和调度；32 项正式任务 wall time 合计 6.0989 小时。

## 实验协议

| 项目 | 本轮标准 |
| --- | --- |
| 参考代码 | Meta SpinQuant，commit `8f47aa3f00e8662caf1a484153920a07e5281c3a`，冻结于 `source/official/` |
| 官方执行方式 | 调用官方模型、旋转、量化和 SGDG 模块的受控适配循环；并非直接运行完整 Hugging Face Trainer |
| TinyLlama | 本地 TinyLlama-1.1B 权重，两端 FP32；训练长度 2048；精确上游模型 revision 尚未确认，以输入 SHA256 锁定身份 |
| Mistral | Mistral-7B-v0.1，两端使用同一 FP16 safetensors；原权重转换已逐张量核对；训练长度 512 |
| 旋转训练 | W16A4KV4，100 steps，单卡，batch=1，梯度累积=1，仅优化旋转矩阵（FP32） |
| 优化与随机数 | LR=1.5，cosine，无 warmup，全局梯度范数裁剪 1.0；初始化/校准 seed=0，训练 RNG 按官方 Trainer 默认重置为 42 |
| 训练输入 | 官方 CustomJsonDataset 逐记录分词拼接，取前 100 个块，不打乱；相同模型两端共享 token 数组和初始矩阵 |
| 固定探针 | 两个未参与旋转优化的 train 块，训练前后保持不变；不用于按 test PPL 选模型 |
| PPL 输入 | WikiText-2 test，文本用双换行连接，无 BOS/EOS，不重叠 2048 token 块，丢弃尾部，batch=1 |
| 测试规模 | TinyLlama 166 块、339802 个预测目标；Mistral 163 块、333661 个预测目标。模型 tokenizer 不同，两个模型不比较 token 哈希 |
| PPL 计算 | 两端统一 FP32 log-softmax/gather 求逐 token NLL，NumPy FP64 累加后 exp；是统一评测适配器，非官方原生逐层 evaluator |
| GPTQ | 仅 TinyLlama；WikiText-2 train，128×2048 校准；blocksize=128，percdamp=0.01，无 act-order，w_groupsize=−1，w_clip 开启 |
| A/K/V | 4-bit 配置采用非对称 A/K/V；K/V 分组按 head_dim，TinyLlama=64，Mistral=128 |
| 性能计时 | PPL 循环计时前另做 2 个测试块 warmup；量化时间包含准备/编译，单次测量 |
| 计算设备 | NVIDIA RTX PRO 6000 Blackwell Server Edition，单卡 |
| 环境 | 官方 torch 2.11.0+cu128 / transformers 4.44.2；Jittor 1.3.8.5 / transformers 5.20.2；numpy 1.26.4 |

补充来源说明：`source/official_compatibility.patch` 是准备阶段随历史配置复制的补丁留档；本轮执行导入 `source/official/` 的干净参考源码，适配逻辑位于 `source/run_full_experiment.py`，没有在本轮自动应用该历史补丁。分析和复现应以实际冻结源码、命令与 protocol.json 为准。

## PPL 对齐

相对差异 = (Jittor PPL / 官方 PPL − 1) × 100%。正号表示 Jittor PPL 更高，负号表示更低；差异小不等于数值过程一致。配置名 W4A16 的 A16 表示关闭 A/K/V 低比特量化，实际浮点工作精度仍按模型为 FP32 / FP16；训练配置 W16 的含义同样是关闭低比特权重量化。


### TinyLlama-1.1B

| 方法 | 配置 | 官方 PPL | Jittor PPL | 相对差异 |
| --- | --- | --- | --- | --- |
| FLOAT | FP32 | 7.970237 | 7.970223 | -0.0002% |
| RTN | W4A16 | 10.694103 | 10.692188 | -0.0179% |
| RTN | W4A4KV4 | 162.726562 | 158.625329 | -2.5203% |
| SpinQuant+RTN | W4A16 | 9.223419 | 9.201868 | -0.2336% |
| SpinQuant+RTN | W4A4KV4 | 11.092414 | 11.099380 | +0.0628% |
| GPTQ | W4A16 | 9.049361 | 9.032441 | -0.1870% |
| GPTQ | W4A4KV4 | 90.454729 | 95.308892 | +5.3664% |
| SpinQuant+GPTQ | W4A16 | 8.254008 | 8.262957 | +0.1084% |
| SpinQuant+GPTQ | W4A4KV4 | 9.025760 | 9.020271 | -0.0608% |

### Mistral-7B

| 方法 | 配置 | 官方 PPL | Jittor PPL | 相对差异 |
| --- | --- | --- | --- | --- |
| FLOAT | FP16 | 5.253507 | 5.253466 | -0.0008% |
| RTN | W4A16 | 6.857751 | 6.860289 | +0.0370% |
| RTN | W4A4KV4 | 496.946932 | 524.041616 | +5.4522% |
| SpinQuant+RTN | W4A16 | 5.655529 | 5.657497 | +0.0348% |
| SpinQuant+RTN | W4A4KV4 | 6.121280 | 6.095274 | -0.4248% |


## SpinQuant 带来的 PPL 改善

同框架、同位宽、同量化方法比较“直接量化”和“学习旋转后量化”。下降率 = 1 − 旋转后 PPL / 直接量化 PPL。这是组合方法的结果，当前实验没有随机旋转基线，不能将全部收益单独归因于“学习”相对于“随机旋转”的增益。


| 模型 | 框架 | 方法 | 配置 | 直接量化 PPL | SpinQuant 后 PPL | PPL 下降 |
| --- | --- | --- | --- | --- | --- | --- |
| tinyllama | official | RTN | W4A16 | 10.694103 | 9.223419 | 13.75% |
| tinyllama | official | RTN | W4A4KV4 | 162.726562 | 11.092414 | 93.18% |
| tinyllama | official | GPTQ | W4A16 | 9.049361 | 8.254008 | 8.79% |
| tinyllama | official | GPTQ | W4A4KV4 | 90.454729 | 9.025760 | 90.02% |
| tinyllama | jittor | RTN | W4A16 | 10.692188 | 9.201868 | 13.94% |
| tinyllama | jittor | RTN | W4A4KV4 | 158.625329 | 11.099380 | 93.00% |
| tinyllama | jittor | GPTQ | W4A16 | 9.032441 | 8.262957 | 8.52% |
| tinyllama | jittor | GPTQ | W4A4KV4 | 95.308892 | 9.020271 | 90.54% |
| mistral | official | RTN | W4A16 | 6.857751 | 5.655529 | 17.53% |
| mistral | official | RTN | W4A4KV4 | 496.946932 | 6.121280 | 98.77% |
| mistral | jittor | RTN | W4A16 | 6.860289 | 5.657497 | 17.53% |
| mistral | jittor | RTN | W4A4KV4 | 524.041616 | 6.095274 | 98.84% |

## 训练与 Loss

下面的“固定探针”使用训练前后完全相同的输入，适合观察这一小份输入上的变化；只有两个块，不能代替完整验证集。每步训练 Loss 用不同 batch，不应直接拿首步和末步 Loss 的升降判断收敛。


| 模型 | 框架 | 记录数 | 平均 batch Loss | 固定探针训练前 | 固定探针训练后 | 探针相对变化 | 训练分钟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tinyllama | official | 100 | 2.207408 | 2.297577 | 2.279348 | -0.7934% | 14.94 |
| tinyllama | jittor | 100 | 2.206670 | 2.314538 | 2.293454 | -0.9110% | 2.90 |
| mistral | official | 100 | 2.045592 | 2.151770 | 2.127687 | -1.1192% | 188.30 |
| mistral | jittor | 100 | 2.046635 | 2.138227 | 2.142264 | +0.1888% | 10.90 |

| 模型 | 配对逐步 Loss MAE | RMSE | Pearson r |
| --- | --- | --- | --- |
| tinyllama | 0.010935 | 0.013858 | 0.998131 |
| mistral | 0.017467 | 0.021669 | 0.997737 |


LR 从第 1 个记录点的 1.5 下降到第 100 个记录点的 0.0003700797，与 100 步 cosine 公式一致。日志记录的是每次更新使用的 LR，最后一次更新对应 step=99，因此不必恰好记录到 0。四组随机探针正交误差约为 10⁻⁶ 或更低。

逐步 Loss 的 Pearson r 很高，部分原因是两端使用相同 batch 序列；它只描述曲线接近程度，不是实现等价性或统计显著性的检验。

![100 步原始训练 Loss 对齐](figures/loss_alignment.png)

![固定探针前后对照](figures/fixed_probe_loss.png)

## 性能记录

以下分别是量化准备阶段秒数、warmup 后 PPL 循环秒数。进程启动、模型加载、warmup 和日志等开销不包含在这两列中，完整时间与 RSS/显存采样见 `performance.csv`。浮点项的“量化准备”仅为配置和准备过程，没有实际低比特权重量化。


| 模型 | 方法 | 配置 | 官方准备 s | Jittor 准备 s | 官方评测 s | Jittor 评测 s |
| --- | --- | --- | --- | --- | --- | --- |
| tinyllama | FLOAT | FP32 | 0.75 | 0.03 | 25.10 | 24.47 |
| tinyllama | RTN | W4A16 | 7.24 | 5.29 | 25.07 | 24.38 |
| tinyllama | RTN | W4A4KV4 | 8.54 | 5.05 | 32.27 | 33.78 |
| tinyllama | SpinQuant+RTN | W4A16 | 17.79 | 25.98 | 25.35 | 27.64 |
| tinyllama | SpinQuant+RTN | W4A4KV4 | 17.37 | 26.73 | 32.63 | 37.67 |
| tinyllama | GPTQ | W4A16 | 184.31 | 1307.25 | 25.07 | 24.35 |
| tinyllama | GPTQ | W4A4KV4 | 186.04 | 1339.40 | 32.27 | 33.79 |
| tinyllama | SpinQuant+GPTQ | W4A16 | 195.65 | 1380.37 | 25.33 | 27.41 |
| tinyllama | SpinQuant+GPTQ | W4A4KV4 | 197.40 | 1392.25 | 32.68 | 36.99 |
| mistral | FLOAT | FP16 | 0.79 | 0.04 | 32.12 | 44.01 |
| mistral | RTN | W4A16 | 35.26 | 27.92 | 32.20 | 42.84 |
| mistral | RTN | W4A4KV4 | 35.42 | 12.66 | 51.69 | 65.98 |
| mistral | SpinQuant+RTN | W4A16 | 119.50 | 190.93 | 32.78 | 53.23 |
| mistral | SpinQuant+RTN | W4A4KV4 | 121.32 | 186.10 | 52.22 | 78.16 |


官方训练开启 gradient checkpointing，Jittor 未实现对应检查点；官方在线旋转和 Jittor 的计算精度、SGDG 的 GPU/CPU 路径也不同。因此训练时间差不能解释为通用框架速度优势。Jittor GPTQ 使用 CPU FP64 累加/分解，与官方 GPU FP32 路径不同，其量化准备明显更慢。PPL 循环吞吐仅描述本轮实现，未进行 INT4 打包和整数 GEMM，不能据此声称低比特推理加速。GPU 峰值是约每 10 秒一次的卡级采样，可能漏掉瞬时峰值；不能视作精确的进程峰值显存。

## 核验与证据


已校验下载的 285 个原始文件 SHA256，并逐项核对 32 项日志 FINAL_RESULT、原始结果 JSON、退出码、配对输入/初始化/测试哈希、400 条逐步 Loss 和 LR、旋转导出数据、全部 Jittor GPTQ 层覆盖。自动核验共 554 项通过。该核验确认记录完整和结果可追溯，不能证明程序不存在任何潜在 bug。


- `../results/`：32 个原始任务 JSON。
- `../logs/`：原始日志、运行命令和退出/资源统计。
- `../rotations/`：四次训练的旋转导出、100 步 Loss CSV；官方还保存 R.bin。
- `../shared/`：相同模型两个框架共用的数据、初始矩阵；`../input_sha256.json` 锁定模型和数据身份。
- `../source/`、`../source_sha256.json`：实际冻结源码与哈希；`../checks/`：环境、转换和训练前核验。
- `analysis_audit.json`、`analysis_data.json`：本次整理的核验明细和全精度分析数据。
- `ppl_alignment.csv`：14 组 PPL 对齐；`rotation_gains.csv`：12 组旋转前后对照。
- `training_summary.csv`、`training_loss_400.csv`、`loss_alignment_metrics.csv`：训练汇总和逐步记录。
- `performance.csv`：32 项性能记录，单位明确。
- `figures/`：4 类图，每类 PNG/PDF/SVG，可用于 PPT。

## 汇报时建议使用的结论

“基于官方 SpinQuant 代码，在单卡和有限训练数据条件下，完成 Jittor 版本的 W16A4KV4 旋转优化，并在 TinyLlama 和 Mistral 上完成 28 项 WikiText-2 PPL 测试。SpinQuant 后的 6 组官方/Jittor PPL 差异均低于 0.43%，100 步训练 Loss 的配对相关系数超过 0.997。直接 W4A4KV4 量化仍存在最高约 5.45% 的框架间差异，数值路径尚未完全一致。”

不要将本轮描述为论文原始规模的全面复现、所有配置严格对齐、训练稳定性已验证或实现了真 INT4 推理。只有 seed=0 一次训练，未估计标准差或置信区间；目前也缺少随机旋转对照和固定旋转交叉评测来进一步分离旋转训练与量化数值误差。新旧实验协议不同，建议本轮作为当前报告主结果，历史结果单独归档。

