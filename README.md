# DeBERTa-V3-Small for Binary Sentiment Classification on SST-2

> Fine-tuning `microsoft/deberta-v3-small` on the GLUE SST-2 benchmark for binary sentiment classification, achieving **94.7% validation accuracy** — matching RoBERTa-Base while using comparable compute, and outperforming BERT-Base (93.5%).

📄 [Read the Paper](./https://github.com/KenOnoderazZz/deberta-sst2-sentiment/blob/main/Evaluating_the_Sufficiency_of_DeBERTa_V3_Small_for_Binary_Sentiment_Classification_on_the_SST_2_Benchmark.pdf) | 🧠 [Hugging Face Model Card](https://huggingface.co/microsoft/deberta-v3-small)

***

## Key Results

| Model | Parameters | SST-2 Accuracy |
|---|---|---|
| BERT-Base (uncased) | 110M | 93.5% |
| **DeBERTa-V3-Small (ours)** | **141.9M** | **94.7%** ✅ |
| RoBERTa-Base | 125M | 94.8% |
| BERT-Large (uncased) | 340M | 94.9% |
| ALBERT-xxlarge | 235M | 95.2% |
| RoBERTa-Large | 355M | 96.4% |

DeBERTa-V3-Small delivers competitive performance while requiring significantly less compute than larger models — **RoBERTa-Large needs 2.5× more parameters for only +1.7% accuracy**.

***

## Confusion Matrix

```
              Predicted
              Negative  Positive
True Negative   405       23
True Positive    23      421
```

Perfectly symmetric errors (23 FP, 23 FN) → balanced sensitivity to both sentiment classes. No class-specific bias.

***

## Pipeline

```
Load SST-2 Dataset → EDA → Load DeBERTa-V3-Small → Tokenize (max 128 tokens)
       ↓
Fine-tune 2 Epochs (Best Checkpoint Saved) → Evaluate on Validation Set
       ↓
Loss Curves → Confusion Matrix → Baseline Comparison → Error Analysis
```

***

## Model Architecture

| Property | Value |
|---|---|
| Model | `microsoft/deberta-v3-small` |
| Encoder Layers | 6 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Total Parameters | ~141.9M |
| Max Sequence Length | 128 tokens |

***

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 2 |
| Weight Decay | 0.01 |
| Scheduler | Linear (no warmup) |
| Mixed Precision | FP16 |
| Hardware | NVIDIA T4 (Google Colab) |

***

## How to Run

```bash
# 1. Clone and install dependencies
git clone https://github.com/KenOnoderazZz/deberta-sst2-sentiment.git
cd deberta-sst2-sentiment
pip install transformers datasets scikit-learn torch

# 2. Run training
python SML_src_deberta_v3_small_sst2_training.py

# 3. Or run interactively via notebook
jupyter notebook SML_src_DeBERTa_V3_Small_SST2_Training.ipynb
```

***

## Dataset

**SST-2 (Stanford Sentiment Treebank)** via [Hugging Face Datasets](https://huggingface.co/datasets/stanfordnlp/sst2):

| Split | Size | Labels |
|---|---|---|
| Train | 67,349 | 44.2% Negative / 55.8% Positive (1.26:1 ratio) |
| Validation | 872 | 0 = Negative, 1 = Positive |
| Test | 1,821 | Hidden (GLUE server) |

Mean sentence length: **12.5 tokens** — 95th percentile at 31 tokens, supporting a 128-token max.

***

## Error Analysis

Misclassified examples (5.3% error rate) fall into three categories:
1. **Sarcasm** — negated positive expressions
2. **Complex negation** — multi-clause constructions
3. **Mixed sentiment** — reviews with both positive and negative signals

These failure modes are endemic to sentiment classification, not model-specific.

***

## Project Info

- **Course:** Statistical Machine Learning
- **Institution:** Institut Teknologi Sepuluh Nopember (ITS), Surabaya
- **Department:** Statistics
- **Authors:** Razaki Andyatama Ken Onodera, Ahmad Adib Syahmie, Fathi Akbar Firmansyah
- **Supervisors:** Dr. Dedy Dwi Prastyo, Dr. T. Dwi Ary Widhianingsih

