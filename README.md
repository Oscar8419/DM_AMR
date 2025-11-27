# 基于扩散模型的调制识别数据增强 (Modu-DM)

这是一个使用条件扩散模型（Conditional Diffusion Model）进行无线电调制信号生成与数据增强的项目。该项目旨在通过生成高质量的合成调制信号来扩充训练数据集，从而提高调制识别（Modulation Recognition）分类器的准确率和鲁棒性。

## 项目简介

在无线通信领域，获取大量标注良好的真实信号数据往往成本高昂。本项目利用深度生成模型——具体来说是去噪扩散概率模型（DDPM）——来学习不同调制类型信号的分布。

核心思想流程如下：

1.  训练一个**条件扩散模型**，使其能够根据类别标签生成特定的调制信号。
2.  训练一个**基线分类器**，用于初步评估和筛选生成的信号。
3.  利用扩散模型生成大量合成信号，并使用基线分类器进行**置信度筛选**，剔除低质量样本。
4.  将高质量的合成信号与原始真实数据混合，形成**增强数据集**。
5.  在增强数据集上训练新的分类器，并验证其性能提升。

## 文件结构

- `config.py`: 项目配置文件。包含数据集路径、超参数（如信号长度、扩散步数、学习率）、训练轮数等。
- `main.py`: 项目的主入口。串联了数据加载、模型训练、信号生成、数据增强和最终评估的完整流程。
- `models.py`: 定义了神经网络模型结构。
  - `ConditionalUNet1D`: 基于 1D UNet 的条件扩散模型，包含 ResBlock 和 AttentionBlock。
  - `CNNClassifier`: 用于调制识别的 1D CNN 分类器。
- `diffusion.py`: 实现了扩散模型的核心逻辑，包括前向加噪过程 (`q_sample`) 和逆向去噪采样过程 (`p_sample`, `sample`)。
- `trainer.py`: 包含扩散模型和分类器的训练循环及评估函数。
- `data_loader.py`: 负责加载和预处理 RadioML 2018.01A 数据集。
- `model_dataset.py`: PyTorch Dataset 定义（如果适用）。

## 环境要求

- Python 3.x
- PyTorch (支持 CUDA 推荐)
- NumPy
- h5py (用于读取 HDF5 数据集)
- tqdm (用于显示进度条)

## 使用方法

1.  **准备数据**:
    下载 RadioML 2018.01A 数据集 (`GOLD_XYZ_OSC.0001_1024.hdf5`)。
    打开 `config.py`，修改 `"data_path"` 为你本地数据集的实际路径。

    ```python
    # config.py
    CONFIG = {
        "data_path": "path/to/your/GOLD_XYZ_OSC.0001_1024.hdf5",
        ...
    }
    ```

2.  **调整参数 (可选)**:
    在 `config.py` 中，你可以调整：

    - `train_data_fraction`: 使用真实数据的比例（例如 1.0 表示使用全部数据，0.1 表示仅使用 10% 的数据来模拟少样本场景）。
    - `num_generated_samples_per_class`: 每类生成的合成样本数量。
    - `confidence_threshold`: 生成样本筛选的置信度阈值。

3.  **运行项目**:
    直接运行 `main.py`：

    ```bash
    python main.py
    ```

## 工作流程详解 (Main Pipeline)

运行 `main.py` 将自动执行以下步骤：

1.  **Step A: 数据加载**
    加载 RadioML 数据集，并根据配置划分训练集和测试集。

2.  **Step B: 扩散模型训练**
    训练 `ConditionalUNet1D` 模型以学习信号分布。训练完成后模型会被保存为 `.pth` 文件，以便下次直接加载。

3.  **Step C: 基线分类器训练**
    仅使用原始训练数据训练一个 `CNNClassifier`。记录其在测试集上的准确率作为基线性能。

4.  **Step D: 生成与筛选 (Data Augmentation)**

    - 使用训练好的扩散模型为每个调制类别生成合成信号。
    - 使用基线分类器对生成信号进行预测。
    - **筛选策略**: 仅保留那些基线分类器预测正确且置信度高于 `confidence_threshold` 的样本。这确保了只有特征明显、质量较高的合成信号被加入训练集。

5.  **Step E: 增强分类器训练**
    将筛选后的合成数据与原始训练数据合并。在新的增强数据集上从头训练一个新的 `CNNClassifier`。

6.  **Step F: 结果评估**
    对比“基线分类器”与“增强分类器”在同一测试集上的准确率，输出性能提升幅度。

## 模型架构

### 扩散模型 (Conditional UNet)

- 采用一维卷积 (Conv1d) 处理时间序列信号。
- 使用正弦位置嵌入 (Sinusoidal Embeddings) 编码扩散时间步 $t$。
- 使用 Embedding 层编码调制类别 $c$，并通过线性层注入到每个残差块中。
- 包含下采样、中间层和上采样路径，并在关键位置使用了自注意力机制 (AttentionBlock) 以捕捉长距离依赖。

### 分类器 (CNN)

- 由三层 1D 卷积层组成，每层后接 ReLU 激活和最大池化。
- 尾部包含全连接层 (Fully Connected Layers) 和 Dropout，用于输出 24 种调制类别的概率。
