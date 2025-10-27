import os
from pathlib import Path

# ===============================================================
# 1. 設定您的資料夾路徑
# 請根據您的實際情況，確認這兩個路徑是正確的
# ===============================================================
# 訓練資料集的根目錄 (包含 violence 和 non-violence 這些分類資料夾)
TRAIN_ROOT = Path(r"C:\program1\Database\Movies\train")

# 測試資料集的根目錄
TEST_ROOT = Path(r"C:\program1\Database\Movies\train_fold_2")


def get_video_folder_names(root_path: Path) -> set:
    """
    遍歷指定的根目錄，找出所有影片的子資料夾名稱。
    
    預期的資料夾結構為：
    root_path / class_name / video_name / (frames...)
    
    Args:
        root_path: 要掃描的根目錄 (訓練集或測試集的路徑)。
        
    Returns:
        一個包含所有影片資料夾名稱的集合 (set)。
    """
    video_names = set()
    
    if not root_path.is_dir():
        print(f"錯誤：找不到路徑 '{root_path}'，請檢查路徑是否正確。")
        return video_names

    # 遍歷第一層子目錄 (分類資料夾，如 'violence', 'non-violence')
    for class_dir in root_path.iterdir():
        if class_dir.is_dir():
            # 遍歷第二層子目錄 (影片資料夾)
            for video_dir in class_dir.iterdir():
                if video_dir.is_dir():
                    # 將影片資料夾的名稱加入集合中
                    video_names.add(video_dir.name)
                    
    return video_names


if __name__ == "__main__":
    print("="*50)
    print("正在開始檢查資料夾重疊情況...")
    print(f"訓練集路徑: {TRAIN_ROOT}")
    print(f"測試集路徑: {TEST_ROOT}")
    print("="*50)

    # 獲取訓練集和測試集的所有影片名稱
    train_video_names = get_video_folder_names(TRAIN_ROOT)
    test_video_names = get_video_folder_names(TEST_ROOT)

    if not train_video_names or not test_video_names:
        print("\n錯誤：一個或多個資料夾為空或路徑錯誤，無法進行比較。")
    else:
        print(f"\n在訓練集中找到 {len(train_video_names)} 個獨立的影片資料夾。")
        print(f"在測試集中找到 {len(test_video_names)} 個獨立的影片資料夾。")
        
        # 使用集合的 'intersection' 功能找出重疊的部分
        common_videos = train_video_names.intersection(test_video_names)
        
        print("\n--- 檢查結果 ---")
        
        if common_videos:
            print(f"警告：發現嚴重的資料洩漏！")
            print(f"共有 {len(common_videos)} 個影片同時存在於訓練集和測試集中。")
            print("重疊的影片資料夾名稱如下：")
            # 為了方便查看，只列出前 20 個
            for i, video_name in enumerate(list(common_videos)[:20]):
                print(f"  - {video_name}")
            if len(common_videos) > 20:
                print("  ...")
                print("  (僅顯示前 20 個)")
        else:
            print("好消息：檢查完成！")
            print("訓練集與測試集的影片資料夾名稱【沒有】直接重疊。")
            print("這排除了最常見的資料洩漏形式。")

    print("\n" + "="*50)