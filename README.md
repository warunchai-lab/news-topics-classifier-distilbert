# 📰 News Topics Classifier (QLoRA & DistilBERT)

An end-to-end NLP text classification project that fine-tunes **DistilBERT** using **4-bit Quantization (QLoRA)** to categorize news articles into four topics: **World**, **Sports**, **Business**, and **Sci/Tech**, deployed via an interactive **Streamlit** web application[cite: 36, 40, 42, 55, 56].

---

## 💡 Project Highlights
* **Dataset:** `fancyzhx/ag_news` (120,000 training samples and 7,600 balanced test samples across 4 classes)[cite: 35, 36, 50].
* **Model Optimization:** Applied 4-bit NormalFloat (`nf4`) quantization (`bitsandbytes`) and LoRA adapters (`peft`), reducing model memory footprint by **33%** (from 251 MB to 169 MB)[cite: 40, 42, 48, 49].
* **Performance Boost:** Improved zero-shot baseline accuracy from **25.00% to 92.64%** (Macro F1: **0.93**) after 3 epochs of fine-tuning[cite: 52].
* **Interactive Inference:** Built a **Streamlit** web interface providing real-time topic predictions and class probability distributions[cite: 55, 56, 59].

---

## 📊 Experimental Results

Four controlled experiments were conducted on **Google Colab (NVIDIA T4 GPU)** to evaluate tokenization casing, training epochs, and LoRA rank ($r$)[cite: 46, 51, 53]:

| Experiment Variant | Tokenizer | LoRA Rank ($r$) | Epochs | Model Size | Baseline Acc. | Fine-Tuned Acc. | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline Uncased | Uncased[cite: 40] | 64[cite: 63] | 1[cite: 63] | 173.3 MB[cite: 49] | 23.99%[cite: 50] | 91.09%[cite: 50] | 0.91[cite: 50] |
| 2. Baseline Cased | Cased[cite: 40] | 64[cite: 42] | 1[cite: 43] | 168.8 MB[cite: 49] | 25.09%[cite: 50] | 91.24%[cite: 50] | 0.91[cite: 50] |
| 3. Higher LoRA Rank | Cased[cite: 62] | 128[cite: 53] | 1[cite: 53] | 168.8 MB[cite: 62] | 20.49%[cite: 54] | 90.99%[cite: 54] | 0.91[cite: 54] |
| **4. Extended Training (Best)** | **Cased** | **64**[cite: 66] | **3**[cite: 66] | **168.8 MB**[cite: 66] | **25.00%**[cite: 52] | **92.64%**[cite: 52] | **0.93**[cite: 52] |

---

## 🗂️ Repository Structure

```text
news-topics-classifier-distilbert/
├── README.md
├── requirements.txt
├── assets/
│   ├── wandb_loss.png
│   └── streamlit_demo.png
├── notebooks/
│   ├── 01_model_comparison_experiments.ipynb
│   └── 02_best_model_3epochs.ipynb
└── app/
    └── app.py
