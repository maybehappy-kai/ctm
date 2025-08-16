# CTM项目数据管理指南——使用符号链接分离代码与数据

## 1. 目的与原则

在共享服务器上进行深度学习项目时，将**代码**与**大型数据文件**（如数据集、模型检查点、日志、实验输出等）分离开来是一种至关重要的最佳实践。

**遵循此原则的主要原因：**

* **存储空间配额 (Storage Quota)**: 用户的个人主目录 (`~/`) 通常有严格的存储空间限制。大型数据集和模型检查点会迅速耗尽此空间。
* **备份策略 (Backup Policy)**: 主目录通常会进行频繁备份。将TB级的、可重新生成的实验数据放在主目录中，会给服务器的备份系统带来巨大且不必要的负担。
* **I/O 性能 (I/O Performance)**: 服务器通常会配置专门的大容量、高性能存储分区（如 `~/Data`），用于密集型读写操作，其性能远超个人主目录。
* **代码整洁与可移植性**: 项目代码库应保持轻量，只包含源代码和配置文件。将数据文件分离可以使代码库的分享、版本控制（如Git）和迁移变得简单高效。

本项目的所有脚本都使用**相对路径**来访问数据和日志文件。为了在不修改任何一行代码的前提下，将数据实际存储在外部目录，我们采用**符号链接 (Symbolic Links)** 的方案。

## 2. 操作流程

以下是在本项目中为Milestone 1设置符号链接的完整示例步骤。后续Milestone如果需要新的输出目录，可以参照此流程进行。

**操作前提**：所有命令都在项目根目录 (`~/ctm/`) 下执行。

### 第一步：在外部数据分区创建目标文件夹

首先，我们在指定的大容量数据目录（此处以 `~/Data` 为例）中，为本项目创建一个总的存储文件夹，并在其中建立分类清晰的子目录。

```bash
# 1. 为本项目创建一个总的存储文件夹
mkdir -p ~/Data/ctm_project_data

# 2. 在总文件夹内，为不同类型的数据创建子目录
mkdir -p ~/Data/ctm_project_data/datasets
mkdir -p ~/Data/ctm_project_data/checkpoints
mkdir -p ~/Data/ctm_project_data/logs
mkdir -p ~/Data/ctm_project_data/M1_outputs
```

### 第二步：准备项目根目录

在创建符号链接之前，我们需要确保项目根目录中没有同名的真实文件夹，以免发生冲突。我们通过重命名的方式来安全地处理已存在的文件夹。

> **注意**: 如果文件夹不存在，以下命令会提示 "No such file or directory"，这是完全正常的，可以安全地忽略。`2>/dev/null || true` 的作用就是抑制这个错误提示并确保脚本继续执行。

```bash
# 安全地重命名项目内可能存在的旧文件夹
mv datasets data_old 2>/dev/null || true
mv checkpoints checkpoints_old 2>/dev/null || true
mv logs logs_old 2>/dev/null || true
mv M1_implementation/M1_outputs M1_implementation/M1_outputs_old 2>/dev/null || true
```

### 第三步：创建符号链接

现在，我们在项目目录中创建指向外部真实数据目录的符号链接。

`ln -s` 命令的格式是 `ln -s /path/to/real_target /path/for/link`。

```bash
# 创建指向数据集的链接
ln -s ~/Data/ctm_project_data/datasets datasets

# 创建指向模型检查点的链接
ln -s ~/Data/ctm_project_data/checkpoints checkpoints

# 创建指向日志的链接
ln -s ~/Data/ctm_project_data/logs logs

# 为Milestone 1的输出创建链接
ln -s ~/Data/ctm_project_data/M1_outputs M1_implementation/M1_outputs
```

## 3. 验证

操作完成后，可以通过 `ls -l` 命令来验证符号链接是否已正确创建。

```bash
ls -l
```

您应该能看到类似下面的输出：

```
lrwxrwxrwx 1 yukaihuang yourgroup   47 Aug 16 17:45  datasets -> /home/yukaihuang/Data/ctm_project_data/datasets
lrwxrwxrwx 1 yukaihuang yourgroup   50 Aug 16 17:45  checkpoints -> /home/yukaihuang/Data/ctm_project_data/checkpoints
lrwxrwxrwx 1 yukaihuang yourgroup   43 Aug 16 17:45  logs -> /home/yukaihuang/Data/ctm_project_data/logs
```
同时，在 `M1_implementation` 文件夹内：
```bash
ls -l M1_implementation/
```
您应该能看到：
```
lrwxrwxrwx 1 yukaihuang yourgroup   48 Aug 16 18:30  M1_outputs -> /home/yukaihuang/Data/ctm_project_data/M1_outputs
```

**关键标识**：
* 行首的第一个字母是 `l`，代表这是一个链接 (link)。
* `->` 符号清晰地指出了链接的名称指向了哪个真实的目标路径。

一旦设置完成，当任何脚本（如 `debug_step.py`）尝试向 `M1_implementation/M1_outputs/` 写入文件时，操作系统会自动将这些文件写入到 `~/Data/ctm_project_data/M1_outputs/`，而我们的Python代码无需感知这一过程。

---

# 实验任务在Milestone中的应用与选择

根据我们制定的方案，具体的实验任务（如Parity、Mazes）从一开始就贯穿始终，但在不同阶段扮演着不同的角色。

---

## Milestone 1 & 2: 实验作为“技术验证平台”

* **何时引入**: **从Milestone 1开始**。
    当我们编写最初的“单步验证”脚本时，就需要一个具体的数据源来驱动一次完整的前向传播。

* **扮演角色**: 在这两个早期阶段，实验任务仅仅是一个**“技术沙箱”**或**“技术验证平台”**。
    * 我们的目标**不是**让模型学会解决这个任务。
    * 我们的目标是利用这个任务产生的数据流，来验证我们的核心架构（`State_t`数据结构、多线程流水线、数据溯源等）在**工程上**是正确、高效且稳定的。

* **推荐选择**: **奇偶校验 (Parity) 任务**。
    * **原因**: Parity任务的数据生成简单、计算开销极小。它能让我们专注于调试代码框架，而不用等待复杂的任务数据处理或模型计算。

---

## Milestone 4 onwards: 实验作为“科学研究对象”

* **何时引入**: **从Milestone 4开始**。
    当我们搭建完成第一个完整的自举系统闭环后，我们的关注点从“代码能否跑通”转向“模型能否学会思考”。

* **扮演角色**: 从这个阶段开始，实验任务转变为我们真正的**“科学研究对象”**。
    * **数据来源 (`k`, `v`)**: 来自于该任务的真实数据。例如，在迷宫任务中，`k`和`v`就来自于迷宫地图的网格表示。
    * **学习目标**: 我们开始观察和评估`StateImprovementScore`，以及模型在该任务上的**最终性能指标**（如迷宫求解成功率）。
    * **科学问题**: 我们开始回答核心问题：“我们的自举系统能否让模型学会解决这个具体问题？”

* **推荐选择**: **迷宫 (Mazes) 任务**。
    * **原因**: 迷宫任务能够直观地展示模型的“思考”轨迹，非常适合分析和验证我们的“无限思考”假设，并且在原始论文中也被证明是CTM的优势所在。

---

### **总结**

* **Milestone 1**: 您需要做的第一个决定就是：“我用哪个任务来搭建和调试我的代码框架？” -> **推荐使用 Parity**。
* **Milestone 4**: 您开始问第一个科学问题：“我的自举系统能否让模型学会解决一个具体问题？” -> **推荐使用 Mazes**。