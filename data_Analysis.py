import os
import re
import numpy as np
from collections import defaultdict

# --- 1. 請修改以下設定 ---

# 來源資料夾設定
root_folder = './runs'
target_filename = 'summary.txt'

# 關鍵字設定 (用於從 summary.txt 提取資料)
accuracy_keyword = 'Accuracy:'
learning_keyword = 'Learning Rate:'
TRAIN_ROOT_keyword = 'TRAIN ROOT:'
TEST_ROOT_keyword = 'TEST ROOT:'
fold_keyword = 'FOLD:'
frames_keyword = 'Frames:'
epochtime_keyword = 'Epoch time:'
totaltime_keyword = 'Total Runtime (sec):'

# 輸出檔案名稱設定
raw_output_filename = 'raw_summary.txt' # 輸出檔案1：原始數據整合
analysis_output_filename = 'statistical_analysis.txt' # 輸出檔案2：統計分析結果

# --- 設定結束 ---


def extract_value(file_path, keyword, numeric_only=False):
    """
    從指定的檔案中，根據關鍵字提取對應的值。
    可選擇是否只提取數字部分。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if keyword in line:
                    value_part = line.split(keyword, 1)[1].strip()
                    if numeric_only:
                        # 只保留數字、小數點和負號
                        numeric_value = ''.join(filter(lambda char: char.isdigit() or char in ['.', '-'], value_part.split()[0]))
                        return numeric_value
                    else:
                        return value_part
    except FileNotFoundError:
        # 這個警告會在主函式中統一處理，這裡回傳 None 即可
        return None
    except Exception as e:
        print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
        return None
    return None


def collect_raw_data():
    """
    第一步：掃描資料夾，收集所有實驗的原始數據。
    """
    print(f"--- 步驟 1: 開始掃描 '{root_folder}' 資料夾並收集原始數據 ---")
    
    all_results = []

    try:
        folder_list = sorted(os.listdir(root_folder))
    except FileNotFoundError:
        print(f" 錯誤: 找不到指定的根目錄 '{root_folder}'。請確保您的 python 程式和 '{root_folder}' 資料夾在同一層。")
        return None

    for folder_name in folder_list:
        subfolder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(subfolder_path):
            txt_file_path = os.path.join(subfolder_path, target_filename)
            
            if not os.path.exists(txt_file_path):
                print(f"  - 警告: 在 {folder_name} 中找不到檔案 {target_filename}，已跳過。")
                continue

            # 提取所有需要的欄位
            accuracy = extract_value(txt_file_path, accuracy_keyword, numeric_only=True)
            learning = extract_value(txt_file_path, learning_keyword, numeric_only=True)
            totaltime = extract_value(txt_file_path, totaltime_keyword, numeric_only=True)
            train_root = extract_value(txt_file_path, TRAIN_ROOT_keyword)
            test_root = extract_value(txt_file_path, TEST_ROOT_keyword)
            frames = extract_value(txt_file_path, frames_keyword, numeric_only=True)
            fold = extract_value(txt_file_path, fold_keyword)
            epochtime = extract_value(txt_file_path, epochtime_keyword)

            # 只要有一個關鍵值存在，就記錄下來
            if any([accuracy, learning, totaltime, train_root, test_root, frames, fold, epochtime]):
                result_line = (
                    f"Folder: {folder_name}, "
                    f"訓練集: {train_root if train_root else 'N/A'}, "
                    f"測試集: {test_root if test_root else 'N/A'}, "
                    f"FOLD: {fold if fold else 'N/A'}, "
                    f"學習率: {learning if learning else 'N/A'}, "
                    f"影格數: {frames if frames else 'N/A'}, "
                    f"Epoch: {epochtime if epochtime else 'N/A'}, "
                    f"準確率: {accuracy if accuracy else 'N/A'}, "
                    f"訓練時間: {totaltime if totaltime else 'N/A'}"
                )
                print(f"  - 成功讀取: {folder_name}")
                all_results.append(result_line)
            else:
                print(f"  - 警告: 在 {folder_name} 的 {target_filename} 中找不到任何指定的關鍵字。")

    # 寫入第一個檔案
    try:
        with open(raw_output_filename, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(result + '\n')
        print(f"步驟 1 完成！原始數據已成功儲存至: {raw_output_filename}\n")
        return all_results
    except Exception as e:
        print(f"寫入檔案 {raw_output_filename} 時發生錯誤: {e}")
        return None


def analyze_and_save_stats(raw_data_lines):
    """
    第二步：分析原始數據，計算統計結果並存檔。
    """
    print(f"--- 步驟 2: 開始分析數據並計算統計結果 ---")

    if not raw_data_lines:
        print("  - 沒有可供分析的數據，已跳過。")
        return

    # 正則表達式，用於從每行字串中解析出需要的欄位
    pattern = re.compile(
        r'訓練集:\s*([\w\-]+).*?'
        r'測試集:\s*([\w\-]+).*?'
        r'學習率:\s*([\d\.e-]+).*?'
        r'影格數:\s*(\d+).*?'
        r'準確率:\s*([\d\.]+).*?'
        r'訓練時間:\s*([\d\.]+)'
    )

    records = []
    for line in raw_data_lines:
        m = pattern.search(line)
        if m:
            train_set, test_set, lr_str, frames_str, acc_str, time_str = m.groups()
            try:
                # 將解析出的字串轉換為對應的數字型態
                record = {
                    "train_set": train_set,
                    "test_set": test_set,
                    "lr": float(lr_str),
                    "frames": int(frames_str),
                    "acc": float(acc_str),
                    "time": float(time_str)
                }
                records.append(record)
            except (ValueError, TypeError) as e:
                print(f"  - 警告: 解析行時發生數值轉換錯誤，已跳過。錯誤: {e}\n    行內容: '{line.strip()}'")

    if not records:
        print(" 沒有成功解析到任何可用資料，請檢查 'raw_summary.txt' 的格式是否符合預期。")
        return

    # 使用 defaultdict 分組計算平均值和樣本數
    summary = defaultdict(lambda: {"acc": [], "time": []})
    for r in records:
        key = (r["train_set"], r["test_set"], r["lr"], r["frames"])
        summary[key]["acc"].append(r["acc"])
        summary[key]["time"].append(r["time"])
    
    # 準備輸出的內容
    output_lines = []
    header = f"{'訓練集':<15}{'測試集':<15}{'學習率':<12}{'影格數':<10}{'平均準確率':<15}{'平均訓練時間(s)':<20}{'樣本數':<8}"
    separator = "-" * 95
    
    output_lines.append(header)
    output_lines.append(separator)
    print("\n統計結果預覽：")
    print(header)
    print(separator)

    for (train_set, test_set, lr, frames), vals in sorted(summary.items()):
        mean_acc = np.mean(vals["acc"])
        mean_time = np.mean(vals["time"])
        n_samples = len(vals["acc"])
        
        result_line = (
            f"{train_set:<15}{test_set:<15}{lr:<12.2e}"
            f"{frames:<10}{mean_acc:<15.4f}{mean_time:<20.2f}{n_samples:<8}"
        )
        output_lines.append(result_line)
        print(result_line)

    # 寫入第二個檔案
    try:
        with open(analysis_output_filename, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\n 步驟 2 完成！統計分析結果已成功儲存至: {analysis_output_filename}")
    except Exception as e:
        print(f" 寫入檔案 {analysis_output_filename} 時發生錯誤: {e}")


def main():
    """
    主程式：執行數據收集與分析
    """
    # 步驟 1: 收集原始數據並存檔
    raw_results = collect_raw_data()
    
    # 步驟 2: 如果步驟 1 成功，則進行數據分析並存檔
    if raw_results is not None:
        analyze_and_save_stats(raw_results)

if __name__ == "__main__":
    main()