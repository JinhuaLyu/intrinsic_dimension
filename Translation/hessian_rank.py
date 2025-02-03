import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import functions  # 假设该模块中定义了 preprocess_function

# --------------------------
# 超参数和设置
# --------------------------
learning_rate = 1e-5
batch_size = 16
num_epochs = 1        # 训练周期设为1
weight_decay = 0

output_dir = f"./t5_checkpoints/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}"
os.makedirs("./results/plots", exist_ok=True)
os.makedirs("./results/csv", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# 选择设备
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------
# 加载数据集和预处理
# --------------------------
dataset = load_dataset("wmt14", "fr-en")
train_data = dataset["train"].shuffle(seed=42).select(range(10000))
val_data   = dataset["validation"].shuffle(seed=42).select(range(1000))

tokenizer = T5Tokenizer.from_pretrained("t5-small")
process = lambda data: data.map(functions.preprocess_function, batched=True, remove_columns=data.column_names)
train_dataset = process(train_data)
val_dataset   = process(val_data)

# --------------------------
# 加载模型和数据整理器
# --------------------------
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

# --------------------------
# Hessian 低秩监测（通过 Callback）
# --------------------------
# 使用训练数据集中前 batch_size 个样本构造 sample batch
sample_items = [train_dataset[i] for i in range(batch_size)]
sample_batch = data_collator(sample_items)
sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

# 使用 PyTorch 2.0 推荐的 torch.func.functional_call
from torch.func import functional_call

class HessianRankCallback(TrainerCallback):
    def __init__(self, layer_name, hessian_step_interval=60, sample_batch=None, device="cpu"):
        """
        参数:
          - layer_name: 要计算 Hessian 的权重参数名称。
          - hessian_step_interval: 每隔多少步计算一次 Hessian（这里设为60步）。
          - sample_batch: 用于计算 Hessian 的示例 batch。
          - device: 运行设备。
        """
        self.layer_name = layer_name
        self.hessian_step_interval = hessian_step_interval
        self.sample_batch = sample_batch
        self.device = device
        self.hessian_info = []  # 存储 (global_step, rank, is_low_rank)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.hessian_step_interval == 0:
            if self.sample_batch is None:
                print("没有提供 sample batch 用于 Hessian 计算。")
                return control

            # 切换到评估模式
            model.eval()

            # 定位目标参数
            target_param = None
            for name, param in model.named_parameters():
                if name == self.layer_name:
                    target_param = param
                    break
            if target_param is None:
                print(f"在模型中未找到参数 {self.layer_name}.")
                return control

            # 定义计算 loss 的函数
            def compute_loss(W):
                # 复制当前参数字典
                params = {n: p for n, p in model.named_parameters()}
                # 替换目标参数为新值 W
                params[self.layer_name] = W
                # 将 sample_batch 作为关键字参数传入
                outputs = functional_call(model, params, args=(), kwargs=self.sample_batch)
                # 如果输出为 dict，则提取 "loss"；否则假设返回 tuple，第一个元素为 loss
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs[0]
                return loss

            # 使用当前目标参数作为初始值，确保其 requires_grad=True
            W = target_param.detach().clone().requires_grad_(True)

            try:
                # 计算 Hessian（关于 W 的二阶导数）
                hessian = torch.autograd.functional.hessian(compute_loss, W)
            except Exception as e:
                print("计算 Hessian 时出错：", e)
                model.train()
                return control

            # 将 Hessian 重塑为二维矩阵（numel x numel）
            hessian_matrix = hessian.reshape(W.numel(), W.numel()).detach().cpu().numpy()
            rank = np.linalg.matrix_rank(hessian_matrix)
            min_dim = min(hessian_matrix.shape)
            is_low_rank = rank < (0.7 * min_dim)
            self.hessian_info.append((state.global_step, rank, is_low_rank))
            print(f"Step {state.global_step}: Hessian rank for {self.layer_name}: {rank} (low_rank: {is_low_rank})")

            # 恢复训练模式
            model.train()
        return control

# 实例化 Hessian callback，指定目标参数名称
layer_name = "decoder.block.5.layer.1.EncDecAttention.k.weight"
hessian_callback = HessianRankCallback(
    layer_name=layer_name,
    hessian_step_interval=60,  # 每60步计算一次 Hessian
    sample_batch=sample_batch,
    device=device,
)

# --------------------------
# 设置 Trainer 和训练参数
# --------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=250,
    save_steps=250,
    logging_steps=500,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[hessian_callback]  # 添加自定义的 Hessian callback
)

trainer.train()

# --------------------------
# 保存日志、损失及 Hessian 低秩信息
# --------------------------
log_history = trainer.state.log_history
train_losses = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
eval_losses  = [(entry["step"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]

# 保存训练和评估损失（文件名包含 "hrank"）
np.savetxt("./results/csv/training_loss_hrank.csv", train_losses, delimiter=",", header="step,training_loss", comments="")
np.savetxt("./results/csv/evaluation_loss_hrank.csv", eval_losses, delimiter=",", header="step,evaluation_loss", comments="")

# 保存 Hessian 低秩信息
with open("./results/csv/hessian_rank_info_hrank.csv", "w") as f:
    f.write("step,rank,is_low_rank\n")
    for step, rank, is_low_rank in hessian_callback.hessian_info:
        f.write(f"{step},{rank},{is_low_rank}\n")

# 绘制并保存训练和评估损失曲线（文件名包含 "hrank"）
plt.figure(figsize=(8, 5))
plt.plot(*zip(*train_losses), label="Training Loss")
plt.plot(*zip(*eval_losses), label="Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(f"Training & Evaluation Loss (LR={learning_rate})")
plt.legend()
plt.grid(True)
plt.savefig("./results/plots/loss_curve_hrank.png")
plt.show()

# 切换到评估模式
model.eval()