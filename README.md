# TD-DLDef 两阶段模型变异版

本次修改将模型生成与变异拆成两个阶段：先用多样性评估和 TS-RCS 生成完整原始模型并保存原始权重，再从该原始模型独立生成 PV+BV+IT+NF、PV+BV+IT+WS、PV+BV+IT+WR 三个变体。详细语义、约束适配和复现路径见 [docs/MUTATION_PIPELINE.md](docs/MUTATION_PIPELINE.md)。

以下安装、运行与历史说明沿用原包。默认执行 Keras 3 的 TensorFlow/PyTorch 双后端。data/ 是压缩包原有历史数据。

## 安装与验证

建议 Python 3.12，使用独立虚拟环境。Windows PowerShell：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/check_environment.py --out results/environment.json
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

原包历史验证安装的 PyTorch 为 2.2.0+cpu；本次修改环境未安装深度学习后端。需要 CPU 轮子时可单独从官方索引安装：

```powershell
.venv/Scripts/python.exe -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

Linux/macOS 使用对应平台可安装的轮子与 `.venv/bin/python`。

## 实验

从项目根目录执行。输出必须使用新文件，脚本不会复用历史缓存，避免配置变化后错误复用旧结果。

```powershell
python scripts/run_ablation_exp2.py --pool keras3.10 --ablation full --models 20 --seed 20261009 --out results/full.json
python scripts/run_ablation_exp2.py --pool keras3.10 --ablation all --models 5 --seed 20261010 --workers 3 --out results/ablation.json
python scripts/analyze_exp2_abl.py results/ablation.json
```

`python main.py` 使用同一权威 CLI，参数相同。`--models` 表示原始模型数量；完整配置下，每个成功构建的原始模型计划生成三个独立变体。原始模型保存在编号目录，变体保存在其 `mutants/nf`、`mutants/ws`、`mutants/wr` 子目录。每个实际执行的变体保存模型结构、两个后端模型、共同的变异后权重、原始权重快照、输入和目标、标准输出/错误、结构化运行状态、逐层输出及梯度、比较证据、模型构建源码和 SHA-256 清单。`observations.jsonl/csv/html` 记录候选异常与排除原因。

运行协议是相同初始状态下的一次训练态前向和反向计算，或至多一轮迭代epochs=1，不执行 optimizer.step。默认 MSE 避免随机目标违反概率损失的定义域。随机掩码、未同步状态和未实现的多张量层输出明确排除。输入dtype允许显式转换；比较前核验转换后的指纹。

## 重放保存案例

```powershell
python scripts/reproduce_case.py --case results/full_cases/keras3.10/full/000001/mutants/nf --out results/replay_000001
```

完整案例的重放验证原证据哈希，复制已有模型和数据到空目录，再运行相同协议，不重新随机生成模型或权重。生成失败案例则使用记录的配置和原始权重快照，重新执行对应的单算子生成过程。`model_source.py` 展示合法 Python 构建代码并恢复权重，`reproduce.py` 是便捷入口。源码定位仅为诊断线索，不自动证明该层是根因。跨机器硬件和框架差异可能改变数值，必须保留实际版本信息。

## 历史数据审计和去重

```powershell
python scripts/audit_results.py --data-dir data --output-dir results/history_audit
```

审计以 JSONL 为原始记录，CSV 仅核对投影，汇总 JSON 单独核验。保留来源文件、行号和原始描述，不将缺失证据当误报。

## 方法与修复

- 本次保持上传包的 TS-RCS 选择实现不变：当前代码按 Beta 采样值的倒数排名。
- DIV 依据当前部分模型的 layer/edge/input-shape/input-rank 四空间反馈；任务累计集合仅用于汇总，多输入模型记录全部前驱。
- 原始模型构造期间不运行 PV/BV/IT；完整原始模型生成后，各层统一评估 PV、BV、IT。
- 超时、环境错误、无效输入、未知异常、工具错误分别记录。普通 exit 1/255/-1 不作为框架崩溃证据。原生崩溃也是待调查观测。
- 同输入/目标/权重与执行配置的完整运行才进入比较。使用名称匹配张量并禁止广播；统一 Keras loss 和单次前向路径。
- 采用 dtype 容差与 MAD 双门槛，float16 的 atol=0.005、rtol=0.05 来自补充材料；极端尺度单独记录。ReLU 边界、pool tie 采取保守整例梯度排除，非逐坐标自动识别。
- `utils/gradient_oracle.py` 提供显式纯函数/同步状态/光滑坐标掩码的可选多步长有限差分检查；不自动应用到任意随机模型。
- LIC/LPC/LSC 默认 `paper_v2` 口径，shape/数值域使用 sigma=5 的有限近似。旧 `legacy_0818` 仅用于解释历史数值。

## 更多问题描述、改动位置、验证证据及统计见随附统计文档和验证目录。

原包历史实测记录见 `VERIFICATION.md`。本次变异流程说明见 `docs/MUTATION_PIPELINE.md`。包内 `validation/backend_final_cases` 可直接重放；新运行汇总可用 `python scripts/summarize_run.py validation/backend_final.json --out results/summary.json` 重新生成。`validation/provenance_audit.json` 解释执行期间仅报告代码更新所造成的两份源码指纹。旧说明及旧成功日志在 `docs/history/` 保留。

历史兼容脚本 scripts/gen_report*、scripts/incons 和 scripts/substract 保留用于原记录追溯，未作为新版结果入口验证。新版统计使用 audit_results.py、summarize_run.py 和 analyze_exp2_abl.py。
