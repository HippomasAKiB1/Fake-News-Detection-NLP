# 📰 Fake News Detection using Natural Language Processing (NLP)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) framework for automated fake news detection. This project evaluates multiple text preprocessing workflows, feature extraction techniques (Bag-of-Words and TF-IDF), and machine learning models (Multinomial Naive Bayes, Logistic Regression, and Support Vector Machines) across two benchmark datasets totaling over **116,000 news articles**.

Developed as the **NLP Mid-Term Project** (Summer 2025–2026, Section D, Group 5) at **American International University-Bangladesh (AIUB)**, Faculty of Science & Technology, Department of Computer Science.

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Group Members & Contributions](#-group-members--contributions)
- [Dataset Overview](#-dataset-overview)
- [Data Preprocessing & NLP Pipeline](#-data-preprocessing--nlp-pipeline)
- [Model Architectures & Vectorization](#-model-architectures--vectorization)
- [Experimental Results](#-experimental-results)
- [Key Insights & Theoretical Discussion](#-key-insights--theoretical-discussion)
- [Repository Structure](#-repository-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Run](#-how-to-run)
- [References](#-references)

---

## 🚀 Executive Summary

Fake news proliferation poses severe threats to digital media integrity. This repository presents a systematic comparative study of traditional NLP pipelines and machine learning classifiers designed to distinguish real news from deliberate disinformation.

### Key Highlights
- **Large-Scale Multi-Dataset Benchmark:** Tested on the **ISOT Fake/Real News Dataset** (44,898 articles) and the **WELFake Dataset** (72,095 articles aggregated from 4 sources).
- **Robust Text Cleaning Engine:** Implements customized regex filters and a **Mojibake decoding map** to eliminate UTF-8/Windows-1252 character corruption, URLs, HTML tags, and social media artifacts.
- **Ablation & Optimization Studies:** Features comparative experiments on **Stemming vs. Lemmatization**, **Binary vs. Multinomial Naive Bayes**, custom **from-scratch Naive Bayes implementation**, **Negation Handling (`NOT_` prefixing)**, and **Cross-Dataset Generalization**.
- **State-of-the-Art Benchmark Results:**
  - **Dataset 1 (ISOT):** **99.63% Accuracy** using Logistic Regression with Bag-of-Words.
  - **Dataset 2 (WELFake):** **96.73% Accuracy** using Linear SVM with TF-IDF.

---

## 👥 Group Members & Contributions

| SL | Student Name | Student ID | Core Contributions & Responsibilities |
|---|---|---|---|
| 1 | **Akib Hasan Pyil** | `23-50531-1` | Implemented BoW & TF-IDF vectorization, Naive Bayes experiments (including from-scratch implementation and Binary NB variant), dataset 1 results discussion, and integrated all pipeline stages into a unified notebook (`nlp_mid_project_group_05.ipynb`). |
| 2 | **Mehjabin Mostafa** | `23-50707-1` | Data acquisition, dataset loading logic (`Fake.csv`, `True.csv`, `WELFake_Dataset.csv`), regex-based text cleaning pipeline, mojibake handling, and authored dataset documentation. |
| 3 | **Arpita Islam** | `23-50712-1` | Designed tokenization, case-folding, stopword removal, and WordNet lemmatization modules. Implemented Logistic Regression classification models. |
| 4 | **Tauhid Sarker** | `23-52773-2` | Stratified train-test split implementation, Support Vector Machine (`LinearSVC`) modeling, hyperparameter management, and authored final report documentation (`nlp_mid_project_group_05.docx`). |

---

## 📊 Dataset Overview

The study leverages two distinct public Kaggle datasets to evaluate both single-source accuracy and cross-source generalization:

| Metric / Attribute | Dataset 1: ISOT News | Dataset 2: WELFake |
|---|---|---|
| **Files** | `Fake.csv` (23,481), `True.csv` (21,417) | `WELFake_Dataset.csv` (72,095) |
| **Total Clean Samples** | **44,898 articles** | **72,095 articles** |
| **Class Distribution** | 52% Fake / 48% Real (Balanced) | 51% Fake / 49% Real (Balanced) |
| **Training Set (80%)** | 35,918 samples | 57,676 samples |
| **Test Set (20%)** | 8,980 samples | 14,419 samples |
| **Data Sources** | Reuters news & flagged fake news websites | Merged from Kaggle, McIntire, Reuters & BuzzFeed Political |
| **Primary Features** | `title`, `text`, `subject`, `date` | `title`, `text`, `label` |

```
Dataset 1 (ISOT):    [ Fake: 23,481 (52.3%) ] [ Real: 21,417 (47.7%) ]
Dataset 2 (WELFake): [ Fake: 37,106 (51.5%) ] [ Real: 34,989 (48.5%) ]
```

---

## ⚙️ Data Preprocessing & NLP Pipeline

The pipeline follows a disciplined multi-stage transformation from raw text strings into normalized numerical feature representations:

```
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│ Raw Article     │ ──> │ Mojibake & Regex    │ ──> │ Lowercase, Stopwords  │
│ (Title + Text)  │     │ Cleaning            │     │ & Lemmatization       │
└─────────────────┘     └─────────────────────┘     └───────────────────────┘
                                                               │
                                                               ▼
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│ Evaluation      │ <── │ Classifier Training │ <── │ BoW & TF-IDF          │
│ Metrics & CM    │     │ (NB / LR / SVM)     │     │ Feature Matrices      │
└─────────────────┘     └─────────────────────┘     └───────────────────────┘
```

### Pipeline Steps:
1. **Content Fusion:** Concatenates `title` and `text` fields into a unified document string, maximizing context availability.
2. **Mojibake Correction:** Replaces corrupted multi-byte sequences resulting from UTF-8 vs. Windows-1252 decoding mismatches (e.g., `â€™` → `'`, `â€œ` → `"`).
3. **Regex Cleaning:**
   - Strips HTTP/HTTPS links and `www` URLs (`https?://\S+|www\.\S+`).
   - Removes HTML markup and entities (`<.*?>|&\w+;`).
   - Eliminates social media handles (`@\w+`) and hashtags (`#\w+`).
   - Filters out non-alphabetic characters (`[^a-zA-Z\s]`).
   - Normalizes whitespace sequences (`\s+`).
4. **Linguistic Normalization:**
   - **Case Folding:** Converts tokens to lowercase.
   - **Stopword Removal:** Eliminates high-frequency NLTK English stopwords.
   - **Single-Character Filtering:** Drops residual single letters (`len(token) > 1`).
   - **Lemmatization:** Uses NLTK `WordNetLemmatizer` to reduce words to standard dictionary forms (preserving real words over harsh stemming roots).

---

## 🔬 Model Architectures & Vectorization

### Feature Representations
- **Bag-of-Words (BoW):** Standard term-frequency matrix generated using `CountVectorizer()`.
  - *Vocabulary Size:* **87,299 features** (Dataset 1), **160,807 features** (Dataset 2).
- **TF-IDF:** Inverse-document-frequency weighted feature matrix via `TfidfVectorizer()`. Down-weights globally frequent words that carry low discriminative power.

### Classification Algorithms
1. **Multinomial Naive Bayes (`MultinomialNB`):** Evaluates conditional word probabilities with Laplace smoothing ($\alpha = 1.0$).
2. **Logistic Regression (`LogisticRegression`):** Linear probabilistic model optimizing log-loss with $L_2$ regularization.
3. **Support Vector Classifier (`LinearSVC`):** Linear margin classifier maximizing decision boundary separation between fake and real news vectors.

---

## 📈 Experimental Results

All models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score** on held-out 20% test sets under identical conditions.

### Consolidated Model Performance Matrix

| Algorithm | Representation | Dataset 1 (ISOT) Accuracy | Dataset 1 Precision | Dataset 1 Recall | Dataset 1 F1 | Dataset 2 (WELFake) Accuracy | Dataset 2 Precision | Dataset 2 Recall | Dataset 2 F1 |
|---|---|---|---|---|---|---|---|---|---|
| **Naive Bayes** | BoW | 95.43% | 0.9565 | 0.9561 | 0.9563 | 89.91% | 0.8989 | 0.9056 | 0.9022 |
| **Naive Bayes** | TF-IDF | 93.84% | 0.9335 | 0.9500 | 0.9416 | 86.82% | 0.8566 | 0.8930 | 0.8744 |
| **Logistic Regression** | BoW | **99.63%** | **0.9966** | 0.9964 | **0.9965** | 96.21% | 0.9508 | 0.9768 | 0.9636 |
| **Logistic Regression** | TF-IDF | 98.90% | 0.9923 | 0.9866 | 0.9894 | 95.10% | 0.9448 | 0.9607 | 0.9527 |
| **Linear SVM** | BoW | 99.61% | 0.9960 | **0.9966** | 0.9963 | 95.42% | 0.9430 | 0.9694 | 0.9560 |
| **Linear SVM** | TF-IDF | 99.55% | 0.9960 | 0.9955 | 0.9957 | **96.73%** | **0.9590** | **0.9781** | **0.9685** |

### Confusion Matrix Highlights (Dataset 1 Top Model: Logistic Regression BoW)
```
                  Predicted Real (0)    Predicted Fake (1)
Actual Real (0)         4268                   16
Actual Fake (1)           17                   4679
```
*Total Errors:* Only **33 out of 8,980 test articles** misclassified (< 0.37% error rate).

---

## 🧠 Key Insights & Theoretical Discussion

1. **The Reuters Dateline Artifact (ISOT Dataset Bias):**
   - Logistic Regression and Linear SVM achieve near-perfect performance (> 99.6%) on Dataset 1 (ISOT).
   - *Cause:* Real articles in ISOT almost universally start with standardized Reuters datelines (e.g., `WASHINGTON (Reuters) -`). Linear models assign strong weights to these opening tokens.
   - *Validation:* On Dataset 2 (WELFake), which merges 4 heterogeneous sources without single-outlet datelines, accuracy drops to **96.73%**, providing a realistic baseline for generic fake news detection.

2. **Naive Bayes vs. Representation Type:**
   - Multinomial Naive Bayes consistently performs **worse with TF-IDF than BoW** (Dataset 1: 95.43% → 93.84%; Dataset 2: 89.91% → 86.82%).
   - *Explanation:* Multinomial NB assumes discrete integer term frequencies. Non-integer TF-IDF weights violate its underlying multinomial probability distribution assumptions.

3. **Linear SVM Performance on Heterogeneous Data:**
   - Linear SVM paired with TF-IDF yields the highest score on Dataset 2 (**96.73%**). Down-weighting frequent corpus-wide words allows SVM to isolate subtle stylistic differences across multi-source data.

4. **Cross-Dataset Generalization Deficit:**
   - Training on Dataset 1 and testing on Dataset 2 (and vice versa) resulted in an accuracy drop to **~81.2%**. This illustrates vocabulary sparsity and domain shift across different temporal contexts and publisher outlets.

---

## 📂 Repository Structure

```
Fake-News-Detection-NLP/
│
├── dataset/                         # Raw Datasets
│   ├── Fake.csv                     # ISOT Fake news articles (23,481 samples)
│   ├── True.csv                     # ISOT Real news articles (21,417 samples)
│   ├── WELFake_Dataset.csv          # WELFake benchmark dataset (72,095 samples)
│   └── README.txt                   # Dataset source documentation
│
├── nlp_mid_project_group_05.ipynb   # Main submission notebook (Complete pipeline & benchmarks)
├── Fake_News_Detection.ipynb        # Exploratory notebook & supplementary experiments
├── nlp_mid_project_group_05.docx    # Formal mid-term project report (MS Word format)
├── nlp_mid_project_group_05.pdf     # Formal mid-term project report (PDF format)
├── .gitignore                       # Git exclusion rules
└── README.md                        # Project documentation (this file)
```

---

## 🛠️ Getting Started & Installation

### Prerequisites
- **Python 3.8+**
- Jupyter Notebook / JupyterLab / VS Code

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HippomasAKiB1/Fake-News-Detection-NLP.git
   cd Fake-News-Detection-NLP
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn nltk jupyter
   ```

4. **Download NLTK Corpora:**
   Run the following snippet in Python to download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

---

## 🖥️ How to Run

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the primary notebook:**
   Select [`nlp_mid_project_group_05.ipynb`](file:///e:/AKiB%27s%20Project%20Book/Fake-News-Detection-NLP/nlp_mid_project_group_05.ipynb).

3. **Execute Notebook Cells:**
   - Run cells sequentially from top to bottom (`Cell` → `Run All`).
   - The notebook automatically loads datasets from `dataset/`, executes regex cleaning, performs lemmatization, trains all 6 classifier combinations, and renders evaluation tables and performance visualization plots.

---

## 📚 References

1. **ISOT Fake and Real News Dataset:** Ahmed H, Traore I, Saad S. *"Detecting fake news headlines with lexical features."* Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
2. **WELFake Dataset:** Shahane S, et al. *"WELFake: Word Embedding-Overlapped Fake News Dataset for Detecting Fake News."* Kaggle: [Fake News Classification (WELFake)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).
3. **Speech and Language Processing (3rd ed.):** Daniel Jurafsky and James H. Martin, 2024.
4. **Natural Language Processing in Action (2nd ed.):** Hobson Lane and Maria Dyshel, 2025.

---

<p align="center"><i>American International University-Bangladesh • Department of Computer Science and Engineering • Natural Language Processing Course (Summer 2025-2026)</i></p>