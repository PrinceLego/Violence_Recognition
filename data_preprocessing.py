import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ==============================================
# 使用者設定
# ==============================================
#data_dir = 'C:/program1/Database/Hockey_Fight' # 請根據您的情況修改此路徑
data_dir = 'C:/program1/Database/Violent_Flows' # 請根據您的情況修改此路徑
#data_dir = 'C:/program1/Database/Movies' # 請根據您的情況修改此路徑
classes = ["Violence", "NonViolence"]
n_folds = 5
random_seed = 42

# 建立一個臨時資料夾來存放所有轉好的影格
all_frames_dir = os.path.join(data_dir, '_all_frames')
os.makedirs(all_frames_dir, exist_ok=True)

# ==============================================
# 函式定義
# ==============================================
def video_to_frames(video_path, output_folder, frame_skip=1):
    """將單一影片轉換為多個幀圖片，如果已存在則跳過"""
    os.makedirs(output_folder, exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
        print(f"  - {os.path.basename(output_folder)} 的影格已存在，跳過轉換。")
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 無法開啟影片: {video_path}")
        return False
        
    count = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_id % frame_skip == 0:
            cv2.imwrite(os.path.join(output_folder, f'frame_{count:04d}.jpg'), frame)
            count += 1
        frame_id += 1
    cap.release()
    print(f"  - {os.path.basename(video_path)} -> {count} frames")
    return True

# ==============================================
# STAGE 1: 將所有影片一次性轉成影格 (不刪除原始檔)
# ==============================================
print("="*50)
print("STAGE 1: 開始將所有影片轉換為影格...")
print("="*50)
frame_folders, frame_labels = [], []

for label_idx, cls_name in enumerate(classes):
    class_path = os.path.join(data_dir, cls_name)
    target_class_path = os.path.join(all_frames_dir, cls_name)
    os.makedirs(target_class_path, exist_ok=True)
    
    if not os.path.isdir(class_path):
        print(f"警告: 找不到類別資料夾 {class_path}，將跳過。")
        continue

    print(f"\n正在處理 '{cls_name}' 類別的影片...")
    video_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.avi', '.mpg'))]
    
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(target_class_path, video_name)
        
        success = video_to_frames(video_path, output_folder)
        
        if success:
            frame_folders.append(output_folder)
            frame_labels.append(label_idx)
            # 🌟 主要修改：此處不再有 os.remove 或 shutil.move 的程式碼

print("\n第一階段完成：所有影片已轉換至中央資料夾 `_all_frames`。")
print("✅ 原始影片檔案已全部保留。\n")

# ==============================================
# STAGE 2: 分割影格資料夾並複製到各 Fold
# ==============================================
print("="*50)
print(f"STAGE 2: 開始將影格資料夾複製至 {n_folds}-Folds...")
print("="*50)

if not frame_folders:
    print("錯誤：沒有找到任何影格資料夾可以進行分割。請檢查 STAGE 1 是否成功執行。")
else:
    frame_folders = np.array(frame_folders)
    frame_labels = np.array(frame_labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(frame_folders, frame_labels)):
        fold_num = fold_idx + 1
        print(f"\n--- Processing Fold {fold_num}/{n_folds} ---")
        
        train_paths, test_paths = frame_folders[train_indices], frame_folders[test_indices]
        train_labels, test_labels = frame_labels[train_indices], frame_labels[test_indices]
        
        print(f"  訓練集數量: {len(train_paths)}")
        print(f"  測試集數量: {len(test_paths)}")
        
        fold_dir = os.path.join(data_dir, f'fold_{fold_num}')
        train_dir = os.path.join(fold_dir, 'train')
        test_dir = os.path.join(fold_dir, 'test')

        for dir_path, paths, labels in [(train_dir, train_paths, train_labels),
                                        (test_dir, test_paths, test_labels)]:
            for src_folder_path, label in zip(paths, labels):
                cls_name = classes[label]
                dest_class_dir = os.path.join(dir_path, cls_name)
                os.makedirs(dest_class_dir, exist_ok=True)
                
                folder_name = os.path.basename(src_folder_path)
                dest_folder_path = os.path.join(dest_class_dir, folder_name)

                if not os.path.exists(dest_folder_path):
                    shutil.copytree(src_folder_path, dest_folder_path)
        
        print(f"  Fold {fold_num} 的影格資料夾已複製完畢。")

    try:
        print("\n正在清理臨時資料夾...")
        shutil.rmtree(all_frames_dir)
        print(f"已成功刪除臨時資料夾: {all_frames_dir}")
    except OSError as e:
        print(f"清理臨時資料夾失敗: {e}")

    print("\n所有 Fold 的資料集已建立完成！")