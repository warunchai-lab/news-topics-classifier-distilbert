# 📰 News Topics Classifier (QLoRA & DistilBERT)

An end-to-end NLP text classification project that fine-tunes **DistilBERT** using **4-bit Quantization (QLoRA)** to categorize news articles into four topics: **World**, **Sports**, **Business**, and **Sci/Tech**, deployed via an interactive **Streamlit** web application.

---

## 💡 Project Highlights
* **Dataset:** `fancyzhx/ag_news` (120,000 training samples and 7,600 balanced test samples across 4 classes).
* **Model Optimization:** Applied 4-bit NormalFloat (`nf4`) quantization (`bitsandbytes`) and LoRA adapters (`peft`), reducing model memory footprint by **33%** (from 251 MB to 169 MB).
* **Performance Boost:** Improved zero-shot baseline accuracy from **25.00% to 92.64%** (Macro F1: **0.93**) after 3 epochs of fine-tuning.
* **Interactive Inference:** Built a **Streamlit** web interface providing real-time topic predictions and class probability distributions.

---

## 📊 Experimental Results

Four controlled experiments were conducted on **Google Colab (NVIDIA T4 GPU)** to evaluate tokenization casing, training epochs, and LoRA rank ($r$):

| Experiment Variant | Tokenizer | LoRA Rank ($r$) | Epochs | Model Size | Baseline Acc. | Fine-Tuned Acc. | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline Uncased | Uncased | 64 | 1 | 173.3 MB | 23.99% | 91.09% | 0.91 |
| 2. Baseline Cased | Cased | 64 | 1 | 168.8 MB | 25.09% | 91.24% | 0.91 |
| 3. Higher LoRA Rank | Cased | 128 | 1 | 168.8 MB | 20.49% | 90.99% | 0.91 |
| **4. Extended Training (Best)** | **Cased** | **64** | **3** | **168.8 MB** | **25.00%** | **92.64%** | **0.93** |

---

## 🗂️ Repository Contents

* `01_distilbert_uncased_1epoch.ipynb` — Baseline QLoRA fine-tuning using `distilbert-base-uncased` (1 epoch).
* `02_distilbert_cased_1epoch.ipynb` — Tokenization casing comparison using `distilbert-base-cased` (1 epoch).
* `03_distilbert_cased_3epochs_best.ipynb` — Best-performing configuration (`distilbert-base-cased`, LoRA $r=64$, 3 epochs) achieving **92.64% accuracy**, including weight merging (`merge_and_unload()`) and model export.
* `04_distilbert_cased_lora_r128.ipynb` — Hyperparameter experiment increasing LoRA rank to $r=128$ (1 epoch).
* `05_colab_streamlit_runner.ipynb` — Interactive Streamlit web application script and Colab tunnel deployment runner.

---

## ⚙️ Environment & Usage

> **Note:** Due to `bitsandbytes` 4-bit CUDA dependencies, these notebooks are configured to run on **Google Colab (Linux + NVIDIA T4 GPU)**.

1. **Model Training:** Open `03_distilbert_cased_3epochs_best.ipynb` in Google Colab with a T4 GPU runtime to reproduce the fine-tuning workflow and save the merged model weights.
2. **Web Application:** Run `05_colab_streamlit_runner.ipynb` in Google Colab to launch the interactive Streamlit UI for real-time news classification.
