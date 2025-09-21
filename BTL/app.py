# app.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from collections import Counter
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================
# Thêm thư viện xử lý mất cân bằng
# ======================
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

sns.set_style("whitegrid")

# ======================
# 1. Tiền xử lý dữ liệu
# ======================
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)     # URL
    s = re.sub(r"\S+@\S+", " ", s)                     # Email
    s = re.sub(r"\+?\d[\d\s\-]{5,}\d", " ", s)         # Số điện thoại
    s = re.sub(r"[^a-z0-9\s]", " ", s)                 # Ký tự đặc biệt
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_prepare(path):
    df = pd.read_csv(path, encoding="utf-8")
    df['text'] = df['sms']  # đồng nhất cột dùng cho pipeline
    df['text_clean'] = df['sms'].apply(clean_text)
    df['len_char'] = df['sms'].apply(lambda x: len(str(x)))
    df['len_word'] = df['sms'].apply(lambda x: len(str(x).split()))
    df['has_number'] = df['sms'].str.contains(r"\d").astype(int)
    df['has_special'] = df['sms'].str.contains(r"[^a-zA-Z0-9\s]").astype(int)
    return df

# ======================
# 2. EDA (hiển thị trong Streamlit)
# ======================
def exploratory_data_analysis_streamlit(df):
    # Phân bố nhãn
    label_counts = df['label'].value_counts().sort_index()
    label_percent = (label_counts / label_counts.sum() * 100).round(2)

    st.subheader("📊 Phân bố nhãn (Count & Percent)")
    label_df = pd.DataFrame({
        "Label": label_counts.index.astype(str),
        "Count": label_counts.values,
        "Percent (%)": label_percent.values
    }).set_index("Label")
    st.table(label_df)

    # Biểu đồ cột
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette="viridis", ax=ax)
    ax.set_title("Số lượng theo nhãn")
    st.pyplot(fig)

    # Biểu đồ bánh
    fig2, ax2 = plt.subplots(figsize=(4,3))
    colors = ["#4CAF50", "#F44336"] if len(label_percent) == 2 else None
    ax2.pie(label_percent.values, labels=label_percent.index.astype(str),
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.axis('equal')
    st.pyplot(fig2)

    # Độ dài tin nhắn
    df['msg_len'] = df['text'].apply(len)
    st.subheader("📈 Phân phối độ dài tin nhắn")
    fig3, ax3 = plt.subplots(figsize=(4,3))
    sns.histplot(df['msg_len'], bins=50, kde=True, ax=ax3)
    st.pyplot(fig3)

    # Boxplot
    st.subheader("📦 Boxplot độ dài theo nhãn")
    fig4, ax4 = plt.subplots(figsize=(4,3))
    sns.boxplot(x='label', y='msg_len', data=df, palette="Set2", ax=ax4)
    st.pyplot(fig4)

    # Top từ phổ biến
    st.subheader("🔤 Top 20 từ phổ biến của Ham (0) và Spam (1)")
    ham_words = " ".join(df[df['label'] == 0]['text_clean'].astype(str)).split()
    spam_words = " ".join(df[df['label'] == 1]['text_clean'].astype(str)).split()
    ham_counts = Counter(ham_words).most_common(20)
    spam_counts = Counter(spam_words).most_common(20)

    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    if ham_counts:
        words_ham, counts_ham = zip(*ham_counts)
        sns.barplot(x=list(counts_ham), y=list(words_ham), palette="Blues_r", ax=ax1)
        ax1.set_title("Top 20 từ - Ham (0)")
    if spam_counts:
        words_spam, counts_spam = zip(*spam_counts)
        sns.barplot(x=list(counts_spam), y=list(words_spam), palette="Reds_r", ax=ax2)
        ax2.set_title("Top 20 từ - Spam (1)")
    st.pyplot(fig5)

    # Tỷ lệ chứa số / ký tự đặc biệt
    st.subheader("🔎 Tỷ lệ chứa số / ký tự đặc biệt theo nhãn")
    st.write(df.groupby('label')[['has_number', 'has_special']].mean().round(4))

    # Báo cáo EDA ngắn
    st.subheader("📑 Báo cáo EDA tự động (tóm tắt)")
    report = []
    if label_percent.min() < 30:
        report.append("⚠️ Dữ liệu bị mất cân bằng nhãn, cân nhắc oversampling/undersampling.")
    else:
        report.append("✅ Dữ liệu phân bố khá cân bằng.")
    report.append("📈 Tin nhắn thường ngắn (<200 ký tự).")
    report.append("📦 Spam có xu hướng dài hơn ham.")
    report.append("🔎 Spam chứa nhiều số/ký tự đặc biệt hơn ham.")
    for r in report:
        st.markdown("- " + r)

# ======================
# 3. Huấn luyện & đánh giá
# ======================
def train_and_evaluate(models, X_train, X_test, y_train, y_test, balance_option=None):
    # Xử lý mất cân bằng
    if balance_option == "Oversampling":
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif balance_option == "Undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif balance_option == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    results = []
    cms = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({"Mô hình": name, "Accuracy": acc, "Precision": prec,
                        "Recall": rec, "F1-score": f1})
        cms[name] = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(results), cms

# ======================
# Main pipeline
# ======================
file_path = "BTL/data/train.csv"#"E:/DangVanThanh/train.csv"
df = load_and_prepare(file_path)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(1,2))
X = tfidf.fit_transform(df['text_clean'])
y = df['label'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Sidebar: chọn cân bằng
st.sidebar.subheader("⚖️ Xử lý mất cân bằng dữ liệu")
balance_option = st.sidebar.selectbox(
    "Chọn phương pháp:",
    ["SMOTE", "Không xử lý", "Oversampling", "Undersampling"],
    index=0  # 👉 mặc định SMOTE
)
balance_option = None if balance_option == "Không xử lý" else balance_option

# Model list
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
results_df, cms = train_and_evaluate(models, X_train, X_test, y_train, y_test, balance_option)

# Lưu TF-IDF và best model
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
best_model_name = results_df.sort_values(by="F1-score", ascending=False).iloc[0]["Mô hình"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

# ======================
# Streamlit UI: 3 tab
# ======================
st.set_page_config(page_title="Spam SMS Detector", layout="centered")
tab1, tab2, tab3 = st.tabs(["📊 Khám phá dữ liệu", "🤖 Kết quả mô hình", "📩 Dự đoán Spam/Ham"])

with tab1:
    st.header("📊 Khám phá dữ liệu (EDA)")
    exploratory_data_analysis_streamlit(df)

with tab2:
    st.header("🤖 Kết quả mô hình")
    sorted_df = results_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    st.dataframe(sorted_df)

    for name, model in models.items():
        st.subheader(f"Confusion Matrix - {name}")
        cm = cms[name]
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham (0)", "Spam (1)"],
                    yticklabels=["Ham (0)", "Spam (1)"],
                    ax=ax_cm)
        st.pyplot(fig_cm)

        if name == "Decision Tree":
            st.subheader("🌳 Cây quyết định")
            fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
            tree.plot_tree(model, filled=True,
                           class_names=["Ham", "Spam"],
                           max_depth=3, fontsize=8, ax=ax_tree)
            st.pyplot(fig_tree)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            st.subheader(f"📈 ROC Curve - {name}")
            fig_roc, ax_roc = plt.subplots(figsize=(4,3))
            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

    # Báo cáo mô hình tốt nhất
    best_model = sorted_df.iloc[0]
    best_name = best_model['Mô hình']
    best_clf = models[best_name]
    cm = cms[best_name]
    tn, fp, fn, tp = cm.ravel()
    roc_auc = None
    if hasattr(best_clf, "predict_proba"):
        y_score = best_clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
    report_data = {
        "Mô hình": [best_name],
        "Accuracy": [f"{best_model['Accuracy']:.2f}"],
        "Precision": [f"{best_model['Precision']:.2f}"],
        "Recall": [f"{best_model['Recall']:.2f}"],
        "F1-score": [f"{best_model['F1-score']:.2f}"],
        "TP": [tp], "TN": [tn], "FP": [fp], "FN": [fn],
        "ROC AUC": [f"{roc_auc:.2f}" if roc_auc else "N/A"]
    }
    st.subheader("🔎 Báo cáo tự động - Mô hình tốt nhất")
    st.table(pd.DataFrame(report_data))

with tab3:
    st.header("📩 Dự đoán Spam/Ham")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("best_model.pkl")
    sms = st.text_area("Nhập tin nhắn:", height=150)
    if st.button("Dự đoán"):
        if sms.strip() == "":
            st.warning("Bạn chưa nhập tin nhắn!")
        else:
            sms_clean = clean_text(sms)
            X_new = tfidf.transform([sms_clean])
            y_pred = model.predict(X_new)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_new).max()
            if int(y_pred) == 1:
                st.error(f"🚨 Spam ({prob*100:.2f}%)" if prob else "🚨 Spam")
            else:
                st.success(f"✅ Ham ({prob*100:.2f}%)" if prob else "✅ Ham")
