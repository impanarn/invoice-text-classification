# Invoice Text Classification with Semantic Enrichment and Data Augmentation

This repository implements the paper **“Data Augmentation With Semantic Enrichment for Deep Learning Invoice Text Classification”** on two public datasets:

- **Kaggle invoice descriptions** – short invoice / product descriptions with categorical labels.  
- **SROIE‑based sector dataset** – merchant/address strings derived from ICDAR‑2019 SROIE receipts, labelled as `FOOD`, `HARDWARE`, `RETAIL`, `STATIONERY`, `OTHER`.

The goal is to study how **semantic enrichment** and **WordNet‑based text augmentation** affect classical (LSVM) and deep models (Bi‑LSTM, BERT) on easy vs. harder invoice‑style text classification tasks.

---

## Project structure

- `notebooks/`  
  - Data preparation: cleaning, semantic enrichment, WordNet augmentation for Kaggle and SROIE.  
  - Modelling: LSVM, Bi‑LSTM, BERT notebooks for each dataset, plus result/plot notebooks.

- `data/` (**ignored in Git**)  
  - Local copies of raw and processed CSVs, e.g. `D3_WNtrain100k.csv`, `D2test.csv`, `D3_sroie_WNtrain10k.csv`.  
  - See the data‑prep notebooks for exact filenames and paths.

---

## Methodology (high level)

1. **Data preparation**  
   - Load raw Kaggle/SROIE data.  
   - Clean text: lowercasing, remove digits/symbols, stop words, month names, and non‑English tokens using WordNet; drop blank rows and single‑instance classes; perform an 85/15 train–test split to create `D2train`/`D2test` pairs for each dataset.
2. **Semantic enrichment**  
   - For each training description, append label‑related words to create an enriched short‑text representation (`D3train`).

3. **Text data augmentation (WordNet)**  
   - EDA‑style synonym replacement using the WordNet lexical database.  
   - Kaggle: build `D3_WNtrain100k.csv` (~100k balanced training instances).  
   - SROIE: build a smaller ~10k augmented train set, oversampling minority sectors (FOOD, STATIONERY) to approximate class balance.

4. **Models**

   - **LSVM**  
     - TF‑IDF features (1‑grams and 2‑grams, limited vocabulary).  
     - Provides the main classical baseline.

   - **Bi‑LSTM**  
     - Keras model with Embedding → Bidirectional LSTM → Dense layers, trained on tokenized/padded sequences (max length ≈ 40, vocabulary ≈ 10k).

   - **BERT** (Kaggle only)  
     - Hugging Face `bert-base-uncased` fine‑tuned on the enriched/augmented Kaggle train set using WordPiece tokenization and fixed‑length sequences.

---

## Environment setup

1. **Create a conda environment**

```bash
conda create -n invoice python=3.10
conda activate invoice
pip install -r requirements.txt
```
Download NLTK resources once

In a Python shell or notebook:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```
## GPU (optional but recommended)

If you have an NVIDIA GPU, install the matching CUDA‑enabled builds of TensorFlow and/or PyTorch by following their official installation guides. This is not required for correctness, but it significantly speeds up Bi‑LSTM and BERT training on the augmented datasets.

---

## Running the pipeline

### 1. Prepare data

1. Open the data‑prep notebooks in `notebooks/`.
2. For **Kaggle** and **SROIE** separately:
   - Load the raw CSVs into the expected locations under `data/`.
   - Run text cleaning to produce `D2train` / `D2test`.
   - Run semantic enrichment to produce `D3train`.
   - Run WordNet augmentation to produce:
     - `D3_WNtrain100k.csv` for Kaggle.
     - The ≈10k WordNet‑augmented train set for SROIE.

### 2. Train models

1. Run the **LSVM** notebook using TF‑IDF features on each enriched/augmented train set.
2. Run the **Bi‑LSTM** notebook to train neural models on the same splits (including the class‑weighted variant for SROIE).
3. Run the **BERT** notebook for Kaggle to fine‑tune `bert-base-uncased` on `D3_WNtrain100k` vs `D2test`.

### 3. Generate tables and figures

1. Use the results notebooks to:
   - Build summary tables for accuracy and macro‑F1 on Kaggle and SROIE.
   - Plot bar charts by model, confusion matrices, and word‑cloud / top‑N word frequency plots for key classes.
2. These figures and tables are the ones referenced in the accompanying report/thesis.
