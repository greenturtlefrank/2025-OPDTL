import os
import re
import matplotlib.pyplot as plt

acc_values = []
sparsity_values = [i / 10 for i in range(1, 11)]  # 0.1 ~ 1.0

for s in sparsity_values:
    filename = f"output_{s:.1f}_0.log"

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        acc_values.append(None)
        continue

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        print(f"Empty file: {filename}")
        acc_values.append(None)
        continue

    # 讀取最後一行
    last_line = lines[-1].strip()

    # 用 regex 抓 "AAcc: X.XXX"
    match = re.search(r"Acc:\s*([0-9.]+)", last_line)
    if match:
        acc = float(match.group(1))
        acc_values.append(acc)
    else:
        print(f"No Acc found in last line of {filename}")
        acc_values.append(None)

# ===== 繪圖 =====
plt.figure(figsize=(8, 5))
plt.plot(sparsity_values, acc_values, marker='o', linestyle='-', linewidth=2)

plt.title("Llmam3 MGSM FFN")
plt.xlabel("density")
plt.ylabel("accuracy")
plt.grid(True)
plt.xticks(sparsity_values)

plt.show()
plt.savefig("llama3_MGSM_ffn.png", dpi=300)
