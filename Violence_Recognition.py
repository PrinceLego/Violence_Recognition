# ===============================================================
# int2d3d_tfonly.py — Fixed Final Version
# ===============================================================

import os
import time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import itertools
import io
import traceback
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

# ===============================================================
# Config
# ===============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str)
# 新增學習率參數
parser.add_argument('--learningrate', type=float) # 給一個預設值
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--momentum', type=float)
parser.add_argument('--num_frames', type=int)
parser.add_argument('--img_size', type=int)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--fold', type=str)

args = parser.parse_args()



DATABASE = args.database 
# 'Hockey_Fight' or 'Violent_Flows' or 'Movies' or 'Combined_Violence'
# 'Combined_Violence_Hockey_Fight' or 'Combined_Violence_Violent_Flows' or 'Combined_Violence_Movies'
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNINGRATE = args.learningrate  # 注意：我們使用 --lr，所以這裡用 args.lr
MOMENTUM = args.momentum
NUM_FRAMES = args.num_frames
IMG_SIZE = args.img_size
NUM_CLASSES = args.num_classes
FOLD = args.fold

BASE=Path("C:/program1/Database")
DATAROOT=BASE/DATABASE


if DATABASE == 'Hockey_Fight'or DATABASE == 'Violent_Flows' or DATABASE == 'Movies' :
    
    TRAIN_ROOT = DATAROOT/FOLD/"train"
    VAL_ROOT   = DATAROOT/FOLD/"val"
    TEST_ROOT  = DATAROOT/FOLD/"test"
    TRAIN_DATASET = DATABASE
    TEST_DATASET = DATABASE

elif DATABASE == 'Combined_Dataset_Hockey_Fight' :
    DATAROOT=BASE/"Combined_Dataset"
    TRAIN_ROOT = DATAROOT/FOLD/"train"
    VAL_ROOT   = DATAROOT/FOLD/"val"
    TEST_ROOT  = DATAROOT/FOLD/"test"
    TEST_ROOT  = TEST_ROOT/"Hockey_Fight"
    TRAIN_DATASET = "Combined_Dataset"
    TEST_DATASET = "Hockey_Fight"

elif DATABASE == 'Combined_Dataset_Violent_Flows' :
    DATAROOT=BASE/"Combined_Dataset"
    TRAIN_ROOT = DATAROOT/FOLD/"train"
    VAL_ROOT   = DATAROOT/FOLD/"val"
    TEST_ROOT  = DATAROOT/FOLD/"test"
    TEST_ROOT  = TEST_ROOT/"Violent_Flows"
    TRAIN_DATASET = "Combined_Dataset"
    TEST_DATASET = "Violent_Flows"

elif DATABASE == 'Combined_Dataset_Movies' :
    DATAROOT=BASE/"Combined_Dataset"
    TRAIN_ROOT = DATAROOT/FOLD/"train"
    VAL_ROOT   = DATAROOT/FOLD/"val"
    TEST_ROOT  = DATAROOT/FOLD/"test"
    TEST_ROOT  = TEST_ROOT/"Movies"
    TRAIN_DATASET = "Combined_Dataset"
    TEST_DATASET = "Movies"


# ===============================================================
# Setup output folder by timestamp
# ===============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("runs", f"{ts}")

BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "final_model.h5")
CSV_LOG_PATH = os.path.join(OUT_DIR, "training_log.csv")
TXT_LOG_PATH = os.path.join(OUT_DIR, "summary.txt")
ROC_PNG = os.path.join(OUT_DIR, "roc_curve.png")
PR_PNG = os.path.join(OUT_DIR, "pr_curve.png")
CM_PNG = os.path.join(OUT_DIR, "confusion_matrix.png")
ERROR_LOG = os.path.join(OUT_DIR, "error.log")

def build_file_list(data_root):
    sets = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
        folders, labels = [], []
        for cls_idx, cls in enumerate(sorted(os.listdir(split_dir))):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for vid in sorted(os.listdir(cls_path)):
                vid_path = os.path.join(cls_path, vid)
                if os.path.isdir(vid_path):
                    folders.append(vid_path)
                    labels.append(cls_idx)
        sets[split] = (folders, labels)
    return sets

# ===============================================================
# Model definition
# ===============================================================
def build_int2d3d_model(num_frames=NUM_FRAMES, img_size=IMG_SIZE, num_classes=NUM_CLASSES, conv3d_filters=512):
    inp = tf.keras.Input(shape=(num_frames, img_size, img_size, 3), name="video_input")
    mobilenet = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    mobilenet_v2 = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # allow fine-tuning (you can freeze layers if needed)
    for layer in mobilenet.layers:
        layer.trainable = True
    for layer in mobilenet_v2.layers:
        layer.trainable = True

    td_mn = tf.keras.layers.TimeDistributed(mobilenet)(inp)
    td_mnv2 = tf.keras.layers.TimeDistributed(mobilenet_v2)(inp)

    conv_proj = lambda x: tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (1, 1), padding="same"))(x)
    td_mn = conv_proj(td_mn)
    td_mnv2 = conv_proj(td_mnv2)

    concat = tf.keras.layers.Concatenate(axis=-1)([td_mn, td_mnv2])

    x = tf.keras.layers.BatchNormalization()(concat)
    x = tf.keras.layers.Conv3D(conv3d_filters, (1, 2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="Int2D3D_M1")


def sorted_frames_in_folder(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return [os.path.join(folder, f) for f in files]

class VideoFrameSequence(tf.keras.utils.Sequence):
    def __init__(self, video_folders, labels, batch_size=8,
                 num_frames=16, img_size=224, shuffle=True):
        self.video_folders = video_folders
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(video_folders))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_folders) / float(self.batch_size))) if len(self.video_folders) > 0 else 0

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X, batch_y = [], []
        for i in batch_indices:
            folder = self.video_folders[i]
            label = self.labels[i]
            frames = sorted_frames_in_folder(folder)
            if len(frames) == 0:
                print(f"Warning: No frames found in {folder}, skipping sample.")
                continue
            N = len(frames)
            if N >= 2 * self.num_frames:
                idxs = list(range(0, N, 2))[:self.num_frames]
            else:
                idxs = np.linspace(0, N - 1, self.num_frames).astype(int).tolist()
            imgs = []
            for fid in idxs:
                img = cv2.imread(frames[fid])
                if img is None:
                    if len(imgs) == 0:
                        raise ValueError(f"Cannot read frame {frames[fid]} and no previous frame to copy.")
                    imgs.append(imgs[-1].copy())
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img.astype("float32") / 255.0)
            while len(imgs) < self.num_frames:
                imgs.append(imgs[-1].copy())
            batch_X.append(np.stack(imgs, axis=0))
            batch_y.append(label)
        if len(batch_X) == 0:
            raise ValueError("Empty batch encountered. Check dataset integrity.")
        batch_X = np.stack(batch_X, axis=0)
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=2)
        return batch_X, batch_y

# ===============================================================
# Plot helper functions
# ===============================================================
def plot_roc(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return roc_auc

def plot_pr(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.step(recall, precision, where='post', label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return ap

def plot_confusion_matrix(cm, classes, save_path, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    start_time = time.time()
    try:
# ===========================================================
# Load datasets from separate folders (train / val / test)
# ===========================================================
        def load_split(split_root):
            """Load video folders and labels from a given directory."""
            if not os.path.exists(split_root):
                print(f"Split path not found: {split_root}")
                return [], []
            folders, labels = [], []
            for cls_idx, cls in enumerate(sorted(os.listdir(split_root))):
                cls_path = os.path.join(split_root, cls)
                if not os.path.isdir(cls_path):
                    continue
                for vid in sorted(os.listdir(cls_path)):
                    vid_path = os.path.join(cls_path, vid)
                    if os.path.isdir(vid_path):
                        folders.append(vid_path)
                        labels.append(cls_idx)
            return folders, labels

        train_folders, train_labels = load_split(TRAIN_ROOT)
        val_folders, val_labels = load_split(VAL_ROOT)
        test_folders, test_labels = load_split(TEST_ROOT)

        # summary info
        ds_info = {
            "train_videos": len(train_folders),
            "val_videos": len(val_folders),
            "test_videos": len(test_folders)
        }
        print("Dataset sizes:", ds_info)
        
        # if validation set empty, skip it
        val_gen = None
        if len(val_folders) > 0:
            val_gen = VideoFrameSequence(val_folders, val_labels, batch_size=BATCH_SIZE,num_frames=NUM_FRAMES, shuffle=False)

        train_gen = VideoFrameSequence(train_folders, train_labels, batch_size=BATCH_SIZE,num_frames=NUM_FRAMES, shuffle=True)
        test_gen = VideoFrameSequence(test_folders, test_labels, batch_size=BATCH_SIZE,num_frames=NUM_FRAMES, shuffle=False) if len(test_folders) > 0 else None

        model = build_int2d3d_model(conv3d_filters=512)
        #model.summary()

        opt = tf.keras.optimizers.SGD(learning_rate=LEARNINGRATE, momentum=MOMENTUM)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy" if val_gen else "accuracy", save_best_only=True, verbose=1),
            #tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if val_gen else "loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss" if val_gen else "loss", patience=5, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.CSVLogger(CSV_LOG_PATH)
        ]
        
        history = model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS, callbacks=callbacks, workers=1, use_multiprocessing=False,verbose=2)
        #


        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"\n Training completed successfully, creating output folder: {OUT_DIR}")

        # Save final model
        model.save(FINAL_MODEL_PATH)

        # ===========================================================
        # Evaluate on test set (with safe fallback for eval_model)
        # ===========================================================
        results = {}
        eval_model = None

        if test_gen and len(test_folders) > 0:
            print("\nEvaluating on test set...")

            # choose eval model: best -> final -> in-memory
            if os.path.exists(BEST_MODEL_PATH):
                print(f"Loading best model from: {BEST_MODEL_PATH}")
                eval_model = tf.keras.models.load_model(BEST_MODEL_PATH)
            elif os.path.exists(FINAL_MODEL_PATH):
                print(f"Best model not found. Loading final model from: {FINAL_MODEL_PATH}")
                eval_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
            else:
                print("No saved model found, using in-memory model.")
                eval_model = model

            # 1) Evaluate (progress bar like training)
            test_loss, test_acc = eval_model.evaluate(test_gen, verbose=1)

            # 2) Predict for analysis (also shows progress)
            print("\nGenerating predictions for analysis...")
            probs = eval_model.predict(test_gen, verbose=1)

            # 3) Collect ground truth and preds
            # Note: concatenating labels from generator (Sequence) yields ordered labels if shuffle=False
            y_true = np.argmax(np.concatenate([y for _, y in test_gen], axis=0), axis=1)
            y_pred = np.argmax(probs, axis=1)
            y_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

            # 4) Compute metrics
            cm = confusion_matrix(y_true, y_pred)
            cr_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
            cr_text = classification_report(y_true, y_pred, digits=4)

            # 5) Draw ROC / PR (guarded)
            roc_auc, ap = None, None
            try:
                roc_auc = plot_roc(y_true, y_prob, ROC_PNG)
                ap = plot_pr(y_true, y_prob, PR_PNG)
            except Exception as e:
                print("ROC/PR computation failed:", e)
                traceback.print_exc()

            # 6) Confusion matrix figure
            plot_confusion_matrix(cm, classes=["NonViolence", "Violence"], save_path=CM_PNG)

            # 7) Save arrays and results
            np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
            np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
            np.save(os.path.join(OUT_DIR, "y_prob.npy"), y_prob)

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
            print("⚠️ No test set found or empty test set; skipping evaluation.")

        # ============================
        # Write summary.txt (human readable)
        # ============================
        total_time = time.time() - start_time
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        model_summary_str = buf.getvalue()

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
            for i in range(len(history.history.get("loss", []))):
                f.write(f"Epoch {i+1:02d}: ")
                for k, v in history.history.items():
                    f.write(f"{k}={v[i]:.4f} ")
                f.write("\n")
            f.write("\n")

            if results:
                f.write("=== Test Evaluation ===\n")
                f.write(f"ROC AUC: {results.get('roc_auc')}\n")
                f.write(f"Average Precision: {results.get('average_precision')}\n")

                if "confusion_matrix" in results:
                    f.write("\nConfusion Matrix:\n")
                    cm = np.array(results["confusion_matrix"])
                    for row in cm:
                        f.write(" ".join(f"{int(v):4d}" for v in row) + "\n")

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

        # Email 通知（示範用途）
        sender_email = "prince111299211@gmail.com"
        receiver_email = "prince11299211@gmail.com"
        password = "yajwadiglsdczzkf"  # 建議改用環境變數
        subject = "程式已完成通知"
        body = f"你的程式已經執行完成！" 

        # --- 1. 準備郵件內容 ---
        # 先建立一個包含基本資訊的列表
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
        # 檢查 results 字典是否有內容 (也就是測試是否有被執行)
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



        # 將所有行組合成一個單一的字串
        body = "\n".join(email_body_lines)


        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())

        print("Email 已發送！")

    except Exception as e:
        # ensure we write an error log and still try to write a minimal summary
        with open(ERROR_LOG, "w", encoding="utf-8") as ef:
            ef.write("Exception during run:\n")
            ef.write(traceback.format_exc())
        print(f"Exception occurred, details written to {ERROR_LOG}")
        raise
