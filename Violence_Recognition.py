# ===============================================================
# int2d3d_tfonly.py — Fixed Final Version
# 檔案名稱與描述，說明這是一個基於 TensorFlow 的模型最終版本。
# ===============================================================

# 載入所需的核心函式庫
import os  # 用於操作檔案系統，如路徑處理、建立資料夾
import time  # 用於計時，計算程式執行總時長
import numpy as np  # 用於高效的數值計算，特別是多維陣列操作
import cv2  # OpenCV 函式庫，用於讀取、處理影像（影片幀）
import tensorflow as tf  # 主要的深度學習框架
import matplotlib.pyplot as plt  # 用於繪製圖表，如 ROC 曲線、混淆矩陣
from datetime import datetime  # 用於取得目前時間，為每次執行的結果建立唯一的資料夾名稱
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report # 從 scikit-learn 載入各種評估指標的計算工具
import itertools  # 用於建立迭代器，在繪製混淆矩陣時特別好用
import io  # 用於在記憶體中處理流數據，這裡用來擷取 model.summary() 的輸出
import traceback  # 用於取得詳細的錯誤追蹤訊息
import smtplib  # 用於發送電子郵件
from email.mime.text import MIMEText  # 用於建立郵件內容
from pathlib import Path  # 提供一個物件導向的介面來處理檔案系統路徑，比 os.path 更現代化

# ===============================================================
# Config (組態設定)
# ===============================================================
# 引入 argparse 函式庫，用於解析從命令列傳入的參數
import argparse
# 建立一個 ArgumentParser 物件，用於後續定義參數
parser = argparse.ArgumentParser()
# 定義 --database 參數，型別為字串，用於指定要使用的資料集名稱
parser.add_argument('--database', type=str)
# 定義 --learningrate 參數，型別為浮點數，用於設定模型的學習率
parser.add_argument('--learningrate', type=float) 
# 定義 --epochs 參數，型別為整數，用於設定訓練的總輪數
parser.add_argument('--epochs', type=int)
# 定義 --batch_size 參數，型別為整數，用於設定每一批次的資料量大小
parser.add_argument('--batch_size', type=int)
# 定義 --momentum 參數，型別為浮點數，用於設定 SGD 優化器的動量值
parser.add_argument('--momentum', type=float)
# 定義 --num_frames 參數，型別為整數，用於指定從每個影片中取樣的幀數
parser.add_argument('--num_frames', type=int)
# 定義 --img_size 參數，型別為整數，用於設定輸入影像的尺寸（高和寬）
parser.add_argument('--img_size', type=int)
# 定義 --num_classes 參數，型別為整數，用於設定分類的類別數量（例如，暴力/非暴力為 2）
parser.add_argument('--num_classes', type=int)
# 定義 --fold 參數，型別為字串，用於指定交叉驗證的特定資料夾（例如 'fold1'）
parser.add_argument('--fold', type=str)
parser.add_argument('--sender_email', type=str)
parser.add_argument('--receiver_email', type=str)
parser.add_argument('--password', type=str)

# 解析從命令列傳入的所有參數
args = parser.parse_args()


# 將解析後的參數賦值給全域變數，方便在程式各處使用
DATABASE = args.database  # 資料集名稱
NUM_EPOCHS = args.epochs  # 訓練總輪數
BATCH_SIZE = args.batch_size  # 批次大小
LEARNINGRATE = args.learningrate  # 學習率
MOMENTUM = args.momentum  # 動量
NUM_FRAMES = args.num_frames  # 每個影片的幀數
IMG_SIZE = args.img_size  # 影像尺寸
NUM_CLASSES = args.num_classes  # 分類數量
FOLD = args.fold  # 交叉驗證的折數


SENDER_EMAIL = args.sender_email  # 交叉驗證的折數
RECEIVER_EMAIL = args.receiver_email  # 交叉驗證的折數
PASSWORD = args.password


# 設定資料集的根目錄
BASE = Path("C:/program1/Database")
# 根據傳入的資料集名稱，組合出完整的資料路徑
DATAROOT = BASE / DATABASE

# 這段是根據傳入的 DATABASE 參數來設定訓練、驗證和測試集的具體路徑
# 這種結構是為了處理 "單一資料集訓練與測試" 和 "組合資料集訓練，單一資料集測試" 的情況

# 如果是單一資料集（Hockey_Fight, Violent_Flows, Movies）
if DATABASE == 'Hockey_Fight' or DATABASE == 'Violent_Flows' or DATABASE == 'Movies':
    # 訓練、驗證、測試集的路徑都在該資料集的 FOLD 資料夾下
    TRAIN_ROOT = DATAROOT / FOLD / "train"
    VAL_ROOT   = DATAROOT / FOLD / "val"
    TEST_ROOT  = DATAROOT / FOLD / "test"
    # 記錄訓練和測試所用的資料集名稱
    TRAIN_DATASET = DATABASE
    TEST_DATASET = DATABASE

# ===============================================================
# Setup output folder by timestamp (根據時間戳建立輸出資料夾)
# ===============================================================
# 取得目前時間，並格式化為 "年年月月日日_時時分分秒秒" 的字串
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# 建立一個名為 "runs" 的資料夾，並在其中以時間戳命名一個新的資料夾，用於存放這次執行的所有結果
OUT_DIR = os.path.join("runs", f"{ts}")

# 定義各種輸出檔案的完整路徑
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.h5")      # 存放驗證集上表現最好的模型
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "final_model.h5")     # 存放訓練結束後的最終模型
CSV_LOG_PATH = os.path.join(OUT_DIR, "training_log.csv")       # 存放訓練過程的日誌（loss, accuracy 等）
TXT_LOG_PATH = os.path.join(OUT_DIR, "summary.txt")            # 存放本次執行的總結報告
ROC_PNG = os.path.join(OUT_DIR, "roc_curve.png")               # 存放 ROC 曲線圖
PR_PNG = os.path.join(OUT_DIR, "pr_curve.png")                 # 存放 Precision-Recall 曲線圖
CM_PNG = os.path.join(OUT_DIR, "confusion_matrix.png")         # 存放混淆矩陣圖
ERROR_LOG = os.path.join(OUT_DIR, "error.log")                 # 如果程式出錯，存放錯誤訊息


# ===============================================================
# Model definition (模型定義)
# ===============================================================
# 定義 Int2D3D 模型結構
def build_int2d3d_model(num_frames=NUM_FRAMES, img_size=IMG_SIZE, num_classes=NUM_CLASSES, conv3d_filters=512):
    # 定義模型的輸入層，形狀為 (幀數, 高, 寬, 顏色通道)
    inp = tf.keras.Input(shape=(num_frames, img_size, img_size, 3), name="video_input")
    
    # 載入預訓練的 MobileNet 和 MobileNetV2 模型作為 2D 特徵提取器，不包含頂部的分類層
    mobilenet = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    mobilenet_v2 = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # 設定這兩個預訓練模型的層為可訓練，以便在我們的任務上進行微調（fine-tuning）
    for layer in mobilenet.layers:
        layer.trainable = True
    for layer in mobilenet_v2.layers:
        layer.trainable = True

    # TimeDistributed 層：將一個普通的層（如 MobileNet）應用到序列的每一個時間步上。
    # 這裡的意思是，對影片的每一幀都獨立執行一次 MobileNet 特徵提取。
    td_mn = tf.keras.layers.TimeDistributed(mobilenet)(inp)
    td_mnv2 = tf.keras.layers.TimeDistributed(mobilenet_v2)(inp)

    # 定義一個 1x1 卷積層，用於專案（或調整）特徵圖的通道數
    conv_proj = lambda x: tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (1, 1), padding="same"))(x)
    # 將兩個 MobileNet 輸出的特徵圖通道數都統一到 512
    td_mn = conv_proj(td_mn)
    td_mnv2 = conv_proj(td_mnv2)

    # 沿著特徵通道的維度，將兩個模型的特徵圖拼接起來
    concat = tf.keras.layers.Concatenate(axis=-1)([td_mn, td_mnv2])

    # 接下來是 3D 處理部分
    x = tf.keras.layers.BatchNormalization()(concat) # 批次標準化，穩定訓練
    # 3D 卷積層，卷積核大小為 (1, 2, 2)，這表示它會在時間維度上滑動 1 幀，在空間維度上滑動 2x2。
    # 這樣可以開始融合時間和空間的資訊。
    x = tf.keras.layers.Conv3D(conv3d_filters, (1, 2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x) # 再次批次標準化
    x = tf.keras.layers.Dropout(0.2)(x) # Dropout 層，隨機丟棄 20% 的神經元，防止過擬合
    
    # 全域平均池化層（3D 版本），將整個 3D 特徵圖壓縮成一個向量
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    
    # 全連接層，進行高階特徵組合
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    
    # 輸出層，使用 softmax 活化函數，輸出屬於各個類別的機率
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # 建立並回傳模型，定義其輸入和輸出
    return tf.keras.Model(inputs=inp, outputs=out, name="Int2D3D_M1")


# 一個輔助函式，用於取得一個資料夾中所有按名稱排序的圖片檔案路徑
def sorted_frames_in_folder(folder):
    # 定義支援的圖片副檔名
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    # 遍歷資料夾，找出所有符合副檔名的檔案
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    # 對檔案名進行排序，確保幀的順序是正確的
    files.sort()
    # 回傳完整的檔案路徑列表
    return [os.path.join(folder, f) for f in files]

# 自訂資料產生器，繼承自 tf.keras.utils.Sequence，這是處理大型資料集的標準做法
class VideoFrameSequence(tf.keras.utils.Sequence):
    # 初始化函式
    def __init__(self, video_folders, labels, batch_size=8,
                 num_frames=16, img_size=224, shuffle=True):
        self.video_folders = video_folders  # 影片幀所在的資料夾路徑列表
        self.labels = labels  # 對應的標籤列表
        self.batch_size = batch_size  # 批次大小
        self.num_frames = num_frames  # 每個影片要讀取的幀數
        self.img_size = img_size  # 影像目標尺寸
        self.shuffle = shuffle  # 是否在每個 epoch 結束後打亂資料順序
        self.indices = np.arange(len(video_folders)) # 建立一個索引陣列
        self.on_epoch_end() # 在初始化時先打亂一次

    # 回傳一個 epoch 中總共有多少個批次
    def __len__(self):
        # 如果資料夾列表不為空，則計算 "總樣本數 / 批次大小" 並向上取整
        return int(np.ceil(len(self.video_folders) / float(self.batch_size))) if len(self.video_folders) > 0 else 0

    # 在每個 epoch 結束時會被自動呼叫
    def on_epoch_end(self):
        # 如果設定為 shuffle，則打亂索引
        if self.shuffle:
            np.random.shuffle(self.indices)

    # 根據批次索引 idx 產生一個批次的資料
    def __getitem__(self, idx):
        # 取得這個批次應該包含的資料索引
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 初始化這個批次的影像資料 (X) 和標籤 (y)
        batch_X, batch_y = [], []
        
        # 遍歷批次中的每一個索引
        for i in batch_indices:
            folder = self.video_folders[i] # 取得影片幀資料夾路徑
            label = self.labels[i] # 取得對應的標籤
            frames = sorted_frames_in_folder(folder) # 取得排序好的所有幀路徑
            
            # 如果資料夾是空的，印出警告並跳過這個樣本
            if len(frames) == 0:
                print(f"Warning: No frames found in {folder}, skipping sample.")
                continue
                
            N = len(frames) # 實際的總幀數
            
            # === 新的幀取樣策略：均勻時間取樣 ===
            # 使用 np.linspace 在影片的所有幀 (從索引 0 到 N-1) 中，
            # 均勻地選出 self.num_frames 個幀的索引。
            # 這完美實現了「總幀數 / 需求幀數 = 間隔」的概念，確保取樣能涵蓋整個影片。
            # .astype(int) 將產生的浮點數索引轉換為整數。
            # .tolist() 將 numpy 陣列轉換為 Python 列表。
            idxs = np.linspace(0, N - 1, self.num_frames).astype(int).tolist()
            
            imgs = [] # 儲存這個影片的幀
            for fid in idxs:
                img = cv2.imread(frames[fid]) # 讀取圖片檔案
                
                # === 新增：錯誤處理 ===
                # 檢查 img 是否為 None (讀取失敗)
                if img is None:
                    # 如果 imgs 列表不是空的，就複製最後一張成功讀取的圖片
                    if len(imgs) > 0:
                        print(f"Warning: Failed to read frame {frames[fid]}. Replicating last frame.")
                        imgs.append(imgs[-1].copy())
                    # 如果連第一張都讀取失敗，這是一個嚴重的資料問題，可以選擇跳過或報錯
                    else:
                        print(f"Error: Failed to read the very first frame of {folder}. Skipping this video.")
                        # 這裡我們選擇跳過這個影片，需要確保 batch 不會因此變空
                        # 為了簡化，我們先用一個黑色畫面代替
                        imgs.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.float32))
                    continue # 進入下一次迴圈

                # 如果讀取成功，才進行後續處理
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 將顏色從 BGR (OpenCV預設) 轉為 RGB
                img = cv2.resize(img, (self.img_size, self.img_size)) # 縮放到指定的尺寸
                imgs.append(img.astype("float32") / 255.0) # 將像素值從 0-255 標準化到 0.0-1.0           
            # 如果因為某些原因（例如影片太短）幀數仍然不夠，就用最後一幀來補足
            while len(imgs) < self.num_frames:
                imgs.append(imgs[-1].copy())
            
            # 將處理好的幀列表和標籤加入到批次資料中
            batch_X.append(np.stack(imgs, axis=0)) # np.stack 將幀列表疊成一個 (num_frames, h, w, c) 的陣列
            batch_y.append(label)
        
        # 如果整個批次都是空的（例如，所有樣本資料夾都沒圖片），拋出錯誤
        if len(batch_X) == 0:
            raise ValueError("Empty batch encountered. Check dataset integrity.")
        
        # 將批次內的所有樣本疊成一個大的 numpy 陣列
        batch_X = np.stack(batch_X, axis=0)
        # 將標籤轉換為 one-hot 編碼，例如 [0, 1] -> [[1, 0], [0, 1]]
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=NUM_CLASSES)
        
        # 回傳一個批次的資料 (X, y)
        return batch_X, batch_y

# ===============================================================
# Plot helper functions (繪圖輔助函式)
# ===============================================================
# 繪製 ROC 曲線
def plot_roc(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score) # 計算 FPR (False Positive Rate) 和 TPR (True Positive Rate)
    roc_auc = auc(fpr, tpr) # 計算 AUC (Area Under the Curve)
    plt.figure() # 建立一個新的圖表
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}") # 繪製 ROC 曲線
    plt.plot([0,1],[0,1],'--') # 繪製對角虛線（隨機猜測的基準）
    plt.xlabel("False Positive Rate") # X 軸標籤
    plt.ylabel("True Positive Rate") # Y 軸標籤
    plt.title("ROC Curve") # 圖表標題
    plt.legend(loc="lower right") # 顯示圖例
    plt.grid(True) # 顯示網格
    plt.savefig(save_path, bbox_inches="tight", dpi=300) # 儲存圖檔
    plt.close() # 關閉圖表，釋放記憶體
    return roc_auc # 回傳 AUC 值

# 繪製 Precision-Recall 曲線
def plot_pr(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score) # 計算 Precision 和 Recall
    ap = average_precision_score(y_true, y_score) # 計算 AP (Average Precision)
    plt.figure()
    plt.step(recall, precision, where='post', label=f"AP = {ap:.4f}") # 繪製 PR 曲線
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return ap # 回傳 AP 值

# 繪製混淆矩陣
def plot_confusion_matrix(cm, classes, save_path, normalize=False, cmap=plt.cm.Blues):
    # 如果 normalize=True，則將混淆矩陣的值轉換為百分比
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) # 加上 1e-12 防止除以零
    plt.figure(figsize=(6,6)) # 設定圖表大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap) # 顯示矩陣圖像
    plt.title("Confusion Matrix")
    plt.colorbar() # 顯示顏色條
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45) # 設定 X 軸刻度標籤
    plt.yticks(tick_marks, classes) # 設定 Y 軸刻度標籤
    
    # 在矩陣的每個格子裡填上數字
    fmt = '.2f' if normalize else 'd' # 設定數字格式
    thresh = cm.max() / 2. if cm.size else 0 # 設定文字顏色切換的閾值
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("True label") # Y 軸標籤
    plt.xlabel("Predicted label") # X 軸標籤
    plt.tight_layout() # 自動調整佈局
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# ===============================================================
# Main (主程式)
# ===============================================================
if __name__ == "__main__":
    start_time = time.time() # 記錄程式開始時間
    try: # 使用 try...except 結構來捕捉執行期間的任何錯誤
# ===========================================================
# Load datasets from separate folders (train / val / test) (從不同資料夾載入資料集)
# ===========================================================
        # 定義一個內部函式，用於從指定的路徑載入資料夾和標籤
        def load_split(split_root):
            """從給定的目錄中載入影片資料夾和標籤"""
            if not os.path.exists(split_root):
                print(f"Split path not found: {split_root}")
                return [], [] # 如果路徑不存在，回傳空列表
            
            folders, labels = [], []
            # 遍歷類別資料夾 (e.g., 'Violence', 'NonViolence')，並用 sorted 確保順序一致
            for cls_idx, cls in enumerate(sorted(os.listdir(split_root))):
                cls_path = os.path.join(split_root, cls)
                if not os.path.isdir(cls_path): continue # 如果不是資料夾，則跳過
                # 遍歷每個類別下的影片資料夾
                for vid in sorted(os.listdir(cls_path)):
                    vid_path = os.path.join(cls_path, vid)
                    if os.path.isdir(vid_path): # 確保這是一個資料夾
                        folders.append(vid_path)
                        labels.append(cls_idx) # 將類別索引 (0, 1, ...) 作為標籤
            return folders, labels

        # 呼叫 load_split 函式分別載入訓練、驗證和測試資料
        train_folders, train_labels = load_split(TRAIN_ROOT)
        val_folders, val_labels = load_split(VAL_ROOT)
        test_folders, test_labels = load_split(TEST_ROOT)

        # 顯示資料集大小的摘要資訊
        ds_info = {
            "train_videos": len(train_folders),
            "val_videos": len(val_folders),
            "test_videos": len(test_folders)
        }
        print("Dataset sizes:", ds_info)
        
        # 建立資料產生器實例
        # 如果驗證集是空的，則 val_gen 為 None
        val_gen = None
        if len(val_folders) > 0:
            val_gen = VideoFrameSequence(val_folders, val_labels, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, shuffle=False)

        train_gen = VideoFrameSequence(train_folders, train_labels, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, shuffle=True)
        # 如果測試集是空的，則 test_gen 為 None
        test_gen = VideoFrameSequence(test_folders, test_labels, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, shuffle=False) if len(test_folders) > 0 else None

        # 建立模型
        model = build_int2d3d_model(conv3d_filters=512)
        # model.summary() # 如果需要，可以取消註解來查看模型結構

        # 設定優化器
        opt = tf.keras.optimizers.SGD(learning_rate=LEARNINGRATE, momentum=MOMENTUM)
        # 編譯模型，指定優化器、損失函數和評估指標
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        # 設定回呼函式 (Callbacks)，用於在訓練過程中執行特定操作
        callbacks = [
            # ModelCheckpoint：在每個 epoch 後，如果監控的指標有改善，就儲存最好的模型
            tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy" if val_gen else "accuracy", save_best_only=True, verbose=1),
            # ReduceLROnPlateau (已註解)：如果監控的指標在 N 個 epochs 內沒有改善，就降低學習率
            #tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if val_gen else "loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            # EarlyStopping：如果監控的指標在 N 個 epochs 內沒有改善，就提早停止訓練
            tf.keras.callbacks.EarlyStopping(monitor="val_loss" if val_gen else "loss", patience=5, restore_best_weights=True, verbose=1),
            # CSVLogger：將每個 epoch 的訓練日誌（loss, accuracy等）儲存到 CSV 檔案
            tf.keras.callbacks.CSVLogger(CSV_LOG_PATH)
        ]
        
        # 開始訓練模型
        history = model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS, callbacks=callbacks, workers=1, use_multiprocessing=False, verbose=2)
        
        # 確保輸出目錄存在，如果不存在則建立
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"\n Training completed successfully, creating output folder: {OUT_DIR}")

        # 儲存訓練完成後的最終模型
        model.save(FINAL_MODEL_PATH)

        # ===========================================================
        # Evaluate on test set (在測試集上評估)
        # ===========================================================
        results = {} # 初始化一個字典來儲存評估結果
        eval_model = None

        # 檢查測試資料產生器是否存在且測試集不為空
        if test_gen and len(test_folders) > 0:
            print("\nEvaluating on test set...")

            # 選擇用於評估的模型：優先使用最佳模型，其次是最終模型，最後是在記憶體中的模型
            if os.path.exists(BEST_MODEL_PATH):
                print(f"Loading best model from: {BEST_MODEL_PATH}")
                eval_model = tf.keras.models.load_model(BEST_MODEL_PATH)
            elif os.path.exists(FINAL_MODEL_PATH):
                print(f"Best model not found. Loading final model from: {FINAL_MODEL_PATH}")
                eval_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
            else:
                print("No saved model found, using in-memory model.")
                eval_model = model

            # 1) 評估模型，得到整體的 loss 和 accuracy
            test_loss, test_acc = eval_model.evaluate(test_gen, verbose=1)

            # 2) 預測所有測試資料，得到每個樣本屬於各類別的機率
            print("\nGenerating predictions for analysis...")
            probs = eval_model.predict(test_gen, verbose=1) 

            # 3) 收集真實標籤和預測結果
            # 從產生器中收集所有真實標籤 (因為 shuffle=False，順序是固定的)
            y_true = np.argmax(np.concatenate([y for _, y in test_gen], axis=0), axis=1)
            y_pred = np.argmax(probs, axis=1) # 取得機率最高的類別作為預測標籤
            # 取得屬於 "正類別" (通常是 'Violence', 索引為 1) 的機率，用於計算 ROC/PR
            y_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

            # 4) 計算詳細的評估指標
            cm = confusion_matrix(y_true, y_pred) # 混淆矩陣
            cr_dict = classification_report(y_true, y_pred, digits=4, output_dict=True) # 分類報告 (字典格式)
            cr_text = classification_report(y_true, y_pred, digits=4) # 分類報告 (文字格式)

            # 5) 繪製 ROC / PR 曲線 (加上 try...except 以防出錯)
            roc_auc, ap = None, None
            try:
                roc_auc = plot_roc(y_true, y_prob, ROC_PNG)
                ap = plot_pr(y_true, y_prob, PR_PNG)
            except Exception as e:
                print("ROC/PR computation failed:", e)
                traceback.print_exc()

            # 6) 繪製混淆矩陣圖
            plot_confusion_matrix(cm, classes=["NonViolence", "Violence"], save_path=CM_PNG)

            # 7) 儲存預測結果，方便未來分析
            np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
            np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
            np.save(os.path.join(OUT_DIR, "y_prob.npy"), y_prob)

            # 將所有評估結果更新到 results 字典中
            results.update({
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "confusion_matrix": cm.tolist(),
                "classification_report": cr_dict,
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "average_precision": float(ap) if ap is not None else None,
                "test_samples": int(len(y_true))
            })

            print(f"\n Test Loss: {test_loss:.4f}")
            print(f" Test Accuracy: {test_acc * 100:.2f}%")
        else:
            print("No test set found or empty test set; skipping evaluation.")

        # ============================
        # Write summary.txt (寫入總結報告)
        # ============================
        total_time = time.time() - start_time # 計算總執行時間
        
        # 使用 io.StringIO 在記憶體中建立一個文字緩衝區，來擷取 model.summary() 的輸出
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        model_summary_str = buf.getvalue()

        # 開啟 summary.txt 檔案並寫入所有執行資訊
        with open(TXT_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("=== Int2D3D Model Training Summary ===\n")
            f.write(f"Timestamp: {ts}\n")
            f.write(f"Output Directory: {OUT_DIR}\n")
            f.write(f"TRAIN ROOT: {TRAIN_DATASET}\n")
            f.write(f"TEST ROOT: {TEST_DATASET}\n\n")
            f.write(f"Train Videos: {len(train_folders)}\n")
            f.write(f"Val Videos: {len(val_folders)}\n")
            f.write(f"Test Videos: {len(test_folders)}\n")
            f.write(f"Epochs: {NUM_EPOCHS}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Learning Rate: {LEARNINGRATE}\n")
            f.write(f"Momentum: {MOMENTUM}\n")
            f.write(f"Total Runtime (sec): {total_time:.2f}\n\n")
            f.write(f"Frames: {NUM_FRAMES}\n")
            f.write(f"Epoch time: {len(history.epoch)}\n")
            f.write(f"FOLD: {FOLD}\n")
            f.write("=== Model Architecture ===\n")
            f.write(model_summary_str + "\n")

            f.write("=== Training History ===\n")
            # 遍歷訓練歷史，寫入每個 epoch 的 loss 和 accuracy
            for i in range(len(history.history.get("loss", []))):
                f.write(f"Epoch {i+1:02d}: ")
                for k, v in history.history.items():
                    f.write(f"{k}={v[i]:.4f} ")
                f.write("\n")
            f.write("\n")

            # 如果有測試結果，則寫入測試評估部分
            if results:
                f.write("=== Test Evaluation ===\n")
                f.write(f"ROC AUC: {results.get('roc_auc')}\n")
                f.write(f"Average Precision: {results.get('average_precision')}\n")

                if "confusion_matrix" in results:
                    f.write("\nConfusion Matrix:\n")
                    cm = np.array(results["confusion_matrix"])
                    for row in cm:
                        f.write(" ".join(f"{int(v):4d}" for v in row) + "\n")

                    # 如果是二分類問題，手動計算 Precision, Recall, F1-score
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        precision = tp / (tp + fp + 1e-12)
                        recall = tp / (tp + fn + 1e-12)
                        f1 = 2 * precision * recall / (precision + recall + 1e-12)
                        acc = (tp + tn) / np.sum(cm)
                        f.write(f"\nPrecision: {precision:.4f}\n")
                        f.write(f"Recall:    {recall:.4f}\n")
                        f.write(f"F1-Score:  {f1:.4f}\n")
                        f.write(f"Accuracy:  {acc:.4f}\n")

                if "classification_report" in results and isinstance(results["classification_report"], dict):
                    f.write("\n\n=== Classification Report (per-class) ===\n")
                    cr_dict = results["classification_report"]
                    for cls_name, vals in cr_dict.items():
                        if isinstance(vals, dict):
                            f.write(f"{cls_name:>10}: " + " ".join([f"{k}={v:.4f}" for k, v in vals.items()]) + "\n")
                    f.write("\n")
                else:
                    f.write("\n\nClassification report not available in dict form.\n")

            f.write(f"All results saved in: {OUT_DIR}\n")
            f.write(f"Confusion Matrix image: {CM_PNG}\n")
            f.write(f"ROC Curve: {ROC_PNG}\n")
            f.write(f"PR Curve: {PR_PNG}\n")
            f.write("\n=== End of Report ===\n")

        print(f"\n Summary written to: {TXT_LOG_PATH}")
        print(f"Confusion Matrix saved to: {CM_PNG}")
        print(f"Total runtime: {total_time:.2f} sec\n")



        # ============================
        # Email Notification (Email 通知)
        # ============================

        subject = "程式已完成通知"

        # --- 準備郵件內容 ---
        # 建立一個列表來逐行組織郵件內容，這樣更容易管理
        email_body_lines = [
            "您的實驗已執行完成！",
            f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"輸出資料夾: {OUT_DIR}\n",

            "=== 實驗設定 ===",
            f"訓練集: {TRAIN_DATASET}",
            f"測試集: {TEST_DATASET}",
            f"影格數: {NUM_FRAMES}",
            f"FOLD: {FOLD}",
            f"學習率 (Learning Rate): {LEARNINGRATE}\n"   
        ]
        
        # 如果 results 字典有內容（即測試已執行）
        if results:
            email_body_lines.append("=== 最終測試結果 ===")
            
            # 安全地取得準確率並格式化為百分比
            test_acc = results.get('test_accuracy')
            acc_str = f"{test_acc * 100:.2f}%" if isinstance(test_acc, float) else "N/A"
            
            # 逐行加入測試指標
            email_body_lines.append(f"測試準確率: {acc_str}")
            email_body_lines.append(f"測試損失: {results.get('test_loss', 'N/A'):.4f}")
            email_body_lines.append(f"ROC AUC: {results.get('roc_auc', 'N/A'):.4f}")
            email_body_lines.append(f"平均精確率 (AP): {results.get('average_precision', 'N/A'):.4f}")
            email_body_lines.append(f"epoch次數: {len(history.epoch)}")
            email_body_lines.append(f"訓練時長(sec): {total_time:.2f}\n")

        # 將所有行組合成一個單一的字串，用換行符連接
        body = "\n".join(email_body_lines)

        # 建立郵件物件
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        # 使用 smtplib 的 SSL 連線到 Gmail 的 SMTP 伺服器
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, PASSWORD) # 登入
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string()) # 發送郵件

        print("Email 已發送！")

    except Exception as e: # 如果 try 區塊中發生任何錯誤
        # 確保在發生錯誤時，也能將錯誤訊息寫入日誌檔
        with open(ERROR_LOG, "w", encoding="utf-8") as ef:
            ef.write("Exception during run:\n")
            ef.write(traceback.format_exc()) # 寫入完整的錯誤追蹤訊息
        print(f"Exception occurred, details written to {ERROR_LOG}")
        raise # 重新拋出異常，讓程式終止並顯示錯誤