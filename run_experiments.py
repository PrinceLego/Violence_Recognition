# 檔案名稱: run_experiments.py
import os
import itertools
import time

# ===============================================================
#               實驗的「總控制台」(Parameter Grid)
# ===============================================================
param_grid = {
    '--database': ["Movies","Violent_Flows","Hockey_Fight"],
    #
    '--epochs': [500],
    '--batch_size': [4],
    '--learningrate': [1e-4, 1e-3, 1e-2],
    #, 1e-3, 1e-2
    '--momentum': [0.9],
    '--num_frames': [16,12,8],
    '--img_size': [224],
    '--num_classes': [2],
    '--fold': ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
    '--sender_email': ["prince111299211@gmail.com"],
    '--receiver_email': ["prince11299211@gmail.com"],
    '--password': ["yajwadiglsdczzkf"]
}

# ===============================================================
#           自動化組合與執行邏輯
# ===============================================================
param_names = list(param_grid.keys())
value_lists = list(param_grid.values())
combinations = list(itertools.product(*value_lists))
total_runs = len(combinations)

print(f"找到 {total_runs} 種參數組合，準備開始執行所有實驗...")
print("-" * 60)

# 用來估算剩餘時間
start_all = time.time()
time_records = []  # 每次執行耗時紀錄

for i, combo in enumerate(combinations, start=1):
    # === 顯示進度 ===
    progress = (i / total_runs) * 100
    run_title = f"RUN #{i}/{total_runs}  ({progress:.1f}%)"
    print(f"\n\n{'='*25} {run_title} {'='*25}")

    # === 建立命令字串 ===
    command = "python Violence_Recognition.py"
    for name, value in zip(param_names, combo):
        command += f" {name} {value}"
        # === 修改部分：檢查是否為密碼，若是則不顯示 ===
        if name == '--password':
            None
        elif name == '--sender_email' or name == '--receiver_email':
            print(f"  - {name.replace('--', ''):<12}: {value}")
        else:
            print(f"  - {name.replace('--', ''):<12}: {value}")


    print("\n[Executing Command]:")
    print(command)
    print("-" * (52 + len(run_title)))

    # === 執行實驗並計時 ===
    start_time = time.time()
    os.system(command)
    elapsed = time.time() - start_time
    time_records.append(elapsed)

    # === 顯示時間估算 ===
    avg_time = sum(time_records) / len(time_records)
    remaining_runs = total_runs - i
    est_remaining = avg_time * remaining_runs

    def format_time(sec):
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec/60:.1f} min"
        else:
            h = sec // 3600
            m = (sec % 3600) / 60
            return f"{int(h)}h {m:.0f}m"

    print(f"本次耗時: {format_time(elapsed)}")
    print(f"平均每組: {format_time(avg_time)}")
    print(f"估計剩餘時間: {format_time(est_remaining)}")

print("\n\n全部實驗執行完畢！")
total_elapsed = time.time() - start_all
print(f"總耗時: {int(total_elapsed//3600)} 小時 {int((total_elapsed%3600)//60)} 分鐘")
