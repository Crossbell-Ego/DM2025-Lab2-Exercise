import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,  
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def print_df_info(name, df):
    print(f"[Info] {name} 概況:")
    print(f" Shape: {df.shape}")
    print(f" Columns: {list(df.columns)}")
    if 'emotion' in df.columns:
        print(f" Emotion 分佈:\n{df['emotion'].value_counts().head().to_string(header=False)}")

if __name__ == '__main__':
    
    # ---------------------------------------------------------
    # 初始設定 (Initial Setup)
    # ---------------------------------------------------------
    print_section("階段 1-0: 環境設定")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    
    # ---------------------------------------------------------
    # 資料讀取 (Data Loading)
    # ---------------------------------------------------------
    print_section("階段 1-1: 資料讀取")
    try:
        # 讀取 JSON 檔案 (Read JSON file)
        with open("final_posts.json", "r", encoding="utf-8") as f:
            data_json = json.load(f)
        
        extracted_data = []
        for entry in data_json:
            post = entry.get("root", {}).get("_source", {}).get("post", {})
            extracted_data.append({
                "id": post.get("post_id"),
                "text": post.get("text")
            })
        df_text_raw = pd.DataFrame(extracted_data)
        
        df_emotion = pd.read_csv("emotion.csv")
        df_id = pd.read_csv("data_identification.csv")
        
        print(f"原始資料讀取成功: (Raw data loaded successfully:)")
        print(f"      - JSON Posts: {len(df_text_raw)} 筆")
        print(f"      - Emotion Labels: {len(df_emotion)} 筆")
        print(f"      - ID Map: {len(df_id)} 筆")

    except FileNotFoundError as e:
        print(f"錯誤: 找不到檔案 {e.filename}。 (Error: File not found)")
        sys.exit(1)

    # 合併資料 (Merge data)
    df_merged = df_id.merge(df_text_raw, on="id", how="left")
    
    train_df_raw = df_merged[df_merged["split"] == "train"].copy()
    test_df_raw = df_merged[df_merged["split"] == "test"].copy()
    
    # 合併情緒標籤 (Merge emotion labels)
    train_df = train_df_raw.merge(df_emotion, on="id", how="left")
    
    # ---------------------------------------------------------
    # 資料前處理 (Data Preprocessing)
    # ---------------------------------------------------------
    print_section("階段 1-2: 資料清潔與前處理")
    
    # 基礎清潔 (Basic cleaning)
    print("執行基礎清潔...")
    train_df["text"] = train_df["text"].fillna("").astype(str).str.strip()
    test_df_raw["text"] = test_df_raw["text"].fillna("").astype(str).str.strip() # Test set 也要清 (Also clean test set)
    train_df.dropna(subset=["emotion"], inplace=True)
    
    initial_count = len(train_df)
    
    # 去除重複資料 (Remove duplicate data)
    train_df.drop_duplicates(subset=["text"], keep="first", inplace=True)
    print(f"去除重複貼文: 移除 {initial_count - len(train_df)} 筆")

    # 異常值偵測 (Outlier detection)
    train_df["text_len"] = train_df["text"].apply(len)
    z_scores = stats.zscore(train_df["text_len"])
    train_df = train_df[np.abs(z_scores) < 3] # 保留 Z-Score < 3 (Keep Z-Score < 3)
    
    print(f"異常值移除後，訓練集剩餘: {len(train_df)} 筆")
    print_df_info("最終訓練集", train_df)

    # Label Mapping
    emotions = sorted(train_df["emotion"].unique())
    label2id = {label: i for i, label in enumerate(emotions)}
    id2label = {i: label for i, label in enumerate(emotions)}
    train_df["label"] = train_df["emotion"].map(label2id)
    
    print(f"Label Map: {label2id}")

    # 分割驗證集 (Split validation set)
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df["label"]
    )
    
    # ---------------------------------------------------------
    # TAPT: 任務適應性預訓練 (Task-Adaptive Pre-training)
    # ---------------------------------------------------------
    print_section("階段 2: TAPT (Task-Adaptive Pre-training)")

    # 準備 TAPT 語料 (Train + Test 的所有文本) (Prepare TAPT corpus)
    tapt_texts = pd.concat([train_df["text"], test_df_raw["text"]]).unique()
    tapt_dataset = Dataset.from_dict({"text": tapt_texts})
    print(f"TAPT 語料庫大小: {len(tapt_texts)} 句獨特文本 (TAPT corpus size: unique texts)")

    # 初始化 Tokenizer (Initialize Tokenizer)
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing TAPT corpus...")
    tokenized_tapt = tapt_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 設定 MLM Data Collator  (Set MLM Data Collator)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 初始化 MLM 模型 (Initialize MLM model)
    model_tapt = DistilBertForMaskedLM.from_pretrained(model_checkpoint)

    # TAPT 訓練參數 (TAPT training parameters)
    tapt_args = TrainingArguments(
        output_dir="./tapt_model",
        overwrite_output_dir=True,
        num_train_epochs=5,          
        per_device_train_batch_size=16,
        save_strategy="no",          
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), 
        report_to="none"
    )

    trainer_tapt = Trainer(
        model=model_tapt,
        args=tapt_args,
        train_dataset=tokenized_tapt,
        data_collator=data_collator,
    )

    print("開始 TAPT 訓練 (MLM)... (Start TAPT training)")
    trainer_tapt.train()
    
    # 評估 Perplexity  (Evaluate Perplexity )
    eval_results = trainer_tapt.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"TAPT 完成! Perplexity: {perplexity:.2f} (TAPT completed!)")

    # 儲存 TAPT 後的模型 (Save TAPT-enhanced model)
    tapt_model_path = "./tapt_finetuned_distilbert"
    trainer_tapt.save_model(tapt_model_path)
    print(f"TAPT 模型已儲存至: {tapt_model_path} (Model saved to:)")

    # 清理記憶體 (Clear memory)
    del model_tapt, trainer_tapt
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 特徵工程 (Feature Engineering)
    # ---------------------------------------------------------
    print_section("階段 3: 傳統特徵工程 (For XGBoost)")
    
    print("建立 TF-IDF 特徵... (Build TF-IDF features)")
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), 
        max_df=0.9, min_df=2, max_features=8000
    )
    
    
    X_train_tfidf = vectorizer.fit_transform(train_split["text"])
    X_val_tfidf = vectorizer.transform(val_split["text"])
    X_test_tfidf = vectorizer.transform(test_df_raw["text"])
    
    print("特徵選擇 (Chi2)... (Feature selection)")
    k_best = min(4000, X_train_tfidf.shape[1])
    selector = SelectKBest(chi2, k=k_best)
    X_train_sel = selector.fit_transform(X_train_tfidf, train_split["label"])
    X_val_sel = selector.transform(X_val_tfidf)
    X_test_sel = selector.transform(X_test_tfidf)
    
    print(f"XGBoost 輸入特徵維度: {X_train_sel.shape} (XGBoost input feature dimensions:)")

    # ---------------------------------------------------------
    # 模型 A: TAPT 強化的 BERT (主模型) (Main Model - TAPT-enhanced BERT)
    # ---------------------------------------------------------
    print_section("階段 4: 監督式微調 (Supervised Fine-tuning)")

    # 準備 Dataset (Prepare Dataset)
    hf_train = Dataset.from_pandas(train_split[["text", "label"]])
    hf_val = Dataset.from_pandas(val_split[["text", "label"]])
    hf_test = Dataset.from_pandas(test_df_raw[["text"]])

    tokenized_train = hf_train.map(tokenize_function, batched=True)
    tokenized_val = hf_val.map(tokenize_function, batched=True)
    tokenized_test = hf_test.map(tokenize_function, batched=True)

    # 載入 TAPT 後的模型  (Load TAPT model )
    print(f"載入 TAPT 預訓練權重: {tapt_model_path} (Load TAPT weights:)")
    model_cls = DistilBertForSequenceClassification.from_pretrained(
        tapt_model_path, 
        num_labels=len(emotions)
    )

    # 設定分類訓練參數 (Set classification training parameters)
    cls_args = TrainingArguments(
        output_dir="./cls_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, predictions, average="macro")}

    trainer_cls = Trainer(
        model=model_cls,
        args=cls_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("開始分類模型微調... (Start classification model fine-tuning)")
    trainer_cls.train()
    
    # 預測 (Prediction)
    print("取得 BERT 預測結果... (Get BERT prediction results)")
    pred_output = trainer_cls.predict(tokenized_test)
    y_pred_bert = np.argmax(pred_output.predictions, axis=1)
    
    val_output = trainer_cls.predict(tokenized_val)
    y_val_bert = np.argmax(val_output.predictions, axis=1)
    
    print(f"BERT (TAPT-Enhanced) Val F1: {f1_score(val_split['label'], y_val_bert, average='macro'):.4f}")

    # ---------------------------------------------------------
    # 模型 B: XGBoost (輔助模型) (Auxiliary Model)
    # ---------------------------------------------------------
    print_section("階段 5: 訓練 XGBoost (特徵輔助) (Train XGBoost)")
    
    xgb_clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=len(emotions),
        random_state=SEED,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_clf.fit(X_train_sel, train_split["label"], eval_set=[(X_val_sel, val_split["label"])], verbose=False)
    
    y_pred_xgb = xgb_clf.predict(X_test_sel)
    y_val_xgb = xgb_clf.predict(X_val_sel)
    print(f"      XGBoost Val F1: {f1_score(val_split['label'], y_val_xgb, average='macro'):.4f}")

    # ---------------------------------------------------------
    # 模式整合 (Ensemble) (Model Ensemble)
    # ---------------------------------------------------------
    print_section("階段 6: 模式整合 (Ensemble)")
    
    y_ensemble = []
    conflict_count = 0
    
    for b, x in zip(y_pred_bert, y_pred_xgb):
        if b == x:
            y_ensemble.append(b)
        else:
            y_ensemble.append(b) 
            conflict_count += 1
            
    print(f"整合完成。BERT 與 XGBoost 不一致筆數: {conflict_count}/{len(y_ensemble)} (Ensemble complete. Mismatches:)")

    # ---------------------------------------------------------
    # 評估與輸出 (Evaluation and Output)
    # ---------------------------------------------------------
    print_section("階段 7: 評估與生成提交檔")
    
    # 混淆矩陣 (基於 BERT) (Confusion matrix - based on BERT)
    cm = confusion_matrix(val_split["label"], y_val_bert)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=emotions, yticklabels=emotions)
    plt.title('Validation Confusion Matrix (TAPT-DistilBERT)')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tapt.png')
    print("混淆矩陣已存: confusion_matrix_tapt.png (Confusion matrix saved:)")
    
    print("\nTAPT-BERT 詳細報告: (Detailed report:)")
    print(classification_report(val_split["label"], y_val_bert, target_names=emotions))
    
    # 生成提交檔 (Generate submission file)
    submission = pd.DataFrame({
        "id": test_df_raw["id"],
        "emotion": [id2label[i] for i in y_ensemble]
    })
    
    output_filename = "submission_tapt_xgboost.csv"
    submission.to_csv(output_filename, index=False)
    print(f"提交檔案已建立: {output_filename} (Submission file created:)")
    print(f"預測分布: (Prediction distribution:)\n{submission['emotion'].value_counts()}")