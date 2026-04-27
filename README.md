# Optimizing Food and Medication Intake Through Machine Learning-Based Interaction Prediction

A machine learning pipeline that predicts drug-drug and drug-food interactions from molecular structure, combining Graph Neural Networks (GNNs), Random Forests, and BioBERT-based NLP. Built as part of a Johns Hopkins University MLMA course final project.

---

## Overview

Managing multiple medications safely is often difficult ad patients and physicians have no unified tool that predicts how drugs interact with each other *and* with food simultaneously. This project builds toward that goal by training models on two datasets and integrating them into a single interactive query interface.

**Two interaction domains are covered:**
- **Drug-Drug Interactions (DDI):** Given two drugs (as SMILES strings or names), predict which of 89 interaction mechanism types applies (Random Forest), or which of the top 5 most common types applies (GNN).
- **Drug-Food Interactions:** Given a drug, predict which food-related behaviors apply (e.g., avoid alcohol, take with food, avoid grapefruit) using a multi-label classifier trained on BioBERT embeddings + Morgan fingerprints.

---

## Repository Structure

```
.
├── Data_Visualization.ipynb          # Dataset loading and exploratory visualization
├── notebook1_updated.ipynb           # DDI models: Random Forest (89-class) + GNN (5-class)
├── notebook2_food_model.ipynb        # Drug-food interaction model (BioBERT + RF)
├── notebook3_updated.ipynb           # Inference UI: interactive query tool + model comparison
└── drug_interaction_models/
    ├── ddi_multiclass_model.pkl                 # Trained 89-class RF model
    ├── ddi_label_encoder.pkl                    # LabelEncoder for RF model
    ├── ddi_label_map.json                       # {id: description} for all 89 DDI types
    ├── ddi_gnn_weights.pt                       # GNN state dict (best CV fold)
    ├── ddi_gnn_config.json                      # GNN architecture config + CV metrics
    ├── ddi_gnn_label_encoder.pkl                # LabelEncoder for GNN (top-5 classes)
    ├── ddi_gnn_label_map.json                   # {0–4: description} for top-5 DDI types
    ├── food_behavior_model.pkl                  # Trained multi-label food behavior classifier
    ├── food_mlb.pkl                             # MultiLabelBinarizer for food categories
    ├── food_lookup.json                         # Drug name → {smiles, interactions, labels} lookup table
    ├── food_categories.json                     # List of 13 food behavior category names
    ├── bert_model_name.txt                      # BioBERT model identifier string
    ├── ddi_class_distribution.png               # Top 20 DDI class frequencies (train set)
    ├── ddi_metrics_bar.png                      # RF model accuracy/F1 by split
    ├── ddi_per_class_f1.png                     # Top 25 per-class F1 scores (RF, test set)
    ├── ddi_gnn_training_curves.png              # GNN loss + val accuracy across 3 CV folds
    ├── food_label_distribution.png              # Food behavior category frequencies
    ├── food_metrics_bar.png                     # Food model F1/accuracy by split
    ├── food_per_class_f1.png                    # Per-class F1 for food behavior labels
    ├── RF_vs_GNN_Confusion Matrices.png         # RF vs. GNN confusion matrices on top 5 DDI classes
    ├── RF vs GNN-Per Class F1.png               # RF vs. GNN Per-class F1 on top 5 DDI classes
```

---

## Notebooks

### `Data_Visualization.ipynb` — Dataset Loading & Dry Run

A preliminary notebook that verifies all datasets can be loaded and previewed before any training begins. Run this first to confirm your environment is set up correctly.

**What it does:**
- Loads the PyTDC DrugBank DDI dataset via `tdc.multi_pred.DDI` and prints the train/val/test split sizes and a sample of drug pairs
- Loads the Kaggle drug-food interaction JSON dataset and displays sample records and food interaction text
- Visualizes basic dataset statistics: class distributions, drug count, text length distributions
- Confirms SMILES strings can be parsed by RDKit and converted to molecular graphs

**Run this if:** you want to verify dataset access before committing to the longer training runs in Notebooks 1 and 2.

---

### `notebook1_updated.ipynb` — Drug-Drug Interaction Models

Trains and evaluates two models on the PyTDC DrugBank DDI dataset and saves all artifacts to Google Drive.

#### Part 1 — Random Forest (89-class)

**Data:** 191,808 drug pairs across 1,706 drugs, loaded directly from PyTDC with its pre-defined train/val/test split.

**Steps:**
1. Load the DrugBank DDI dataset via PyTDC and fetch the official 89-class label map
2. Visualize the top 20 interaction types by training frequency
3. Encode integer labels with `LabelEncoder` and handle unseen labels in val/test sets safely
4. Build pair-level Morgan fingerprint features: each drug is encoded as a 2048-bit fingerprint, then the two fingerprints are concatenated into a 4096-dimensional feature vector
5. Train a `RandomForestClassifier` (300 trees, max depth 25, balanced class weights) on the full training split
6. Evaluate on train/val/test: reports accuracy, macro F1, micro F1, and weighted F1
7. Plot per-class F1 for the top 25 classes and a confusion matrix for the top 15 classes

#### Part 2 — Graph Neural Network (5-class)

**Data:** Same DrugBank dataset, filtered to the 5 most frequent interaction types, balanced to 500 samples per class (2,500 pairs total).

**Steps:**
1. Build a balanced subsample from the full dataset (train + val + test pooled, then re-split via cross-validation)
2. Convert each SMILES string to a PyTorch Geometric `Data` object: 12-dimensional node features (9 atom-type one-hot + other flag + degree + aromaticity) plus a 128-bit Morgan fingerprint as an auxiliary graph-level feature
3. Define a 3-layer `GCNConv` encoder (`MolGNN`) that applies message passing, global mean pools to a 64-dim graph embedding, then concatenates the 128-bit fingerprint and projects to a 128-dim molecular embedding
4. Define `DDIGNNModel` which encodes both drugs, concatenates their embeddings, and classifies via a 2-layer MLP with dropout
5. Train with 3-fold stratified cross-validation (50 epochs per fold, Adam optimizer with weight decay, StepLR scheduler)
6. Track and plot training loss and validation accuracy per fold; save the best-performing fold's weights
7. Print a full classification report on the final fold
8. Saved to Drive: GNN weights (`.pt`), config JSON, both label encoders, both label maps, and all figures.
---

### `notebook2_food_model.ipynb` — Drug-Food Interaction Model

Builds a multi-label food behavior classifier using the Kaggle drug-food interaction dataset.

**Data:** 1,423 drugs with free-text food interaction descriptions from the `shayanhusain/drug-food-interactions-dataset` Kaggle dataset.

**Steps:**
1. Download the Kaggle JSON dataset via `kagglehub` and flatten it to a DataFrame (one row per drug)
2. Apply a regex-based rule system to map free-text food interaction descriptions into 13 structured behavior categories:
   `avoid_alcohol`, `avoid_grapefruit`, `take_with_food`, `take_on_empty_stomach`, `avoid_dairy`, `avoid_high_fat_meals`, `avoid_vitamin_k_foods`, `avoid_tyramine_foods`, `avoid_herbal_supplements`, `limit_caffeine`, `increase_fluid_intake`, `avoid_salt_substitutes`, `general_dietary_warning`
3. Look up SMILES strings for all 1,423 drug names via the PubChem API (`pubchempy`); build and save a complete drug → SMILES + labels lookup table (`food_lookup.json`)
4. Load BioBERT (`dmis-lab/biobert-base-cased-v1.2`) and encode each drug's food interaction text into a 768-dimensional embedding using mean pooling over non-padding tokens
5. Compute 2048-bit Morgan fingerprints for all drugs with resolved SMILES; concatenate fingerprints and BERT embeddings into a 2816-dimensional feature vector per drug
6. Binarize multi-labels with `MultiLabelBinarizer`; split into 70% train / 15% val / 15% test
7. Train an `OneVsRestClassifier` wrapping a `RandomForestClassifier` (200 trees, balanced weights) for multi-label prediction
8. Evaluate with micro F1, macro F1, sample F1, and exact match accuracy; plot per-class F1 and a label co-occurrence heatmap
9. Save all model artifacts, the lookup table, and figures to Google Drive
---

### `notebook3_updated.ipynb` — Interactive Query Tool & Model Comparison

Loads all saved models from Google Drive and provides two things: an interactive widget UI for querying drug interactions, and a head-to-head model comparison section.

**Prerequisites:** Notebooks 1 and 2 must be run first so all model artifacts exist in Drive at `drug_interaction_models/`.

#### Module A — DDI Random Forest
Loads `ddi_multiclass_model.pkl` and its label encoder. Given two drug SMILES strings, predicts the top-k most probable interaction types from the 89-class model with confidence scores.

#### Module B — DDI GNN
Rebuilds the `MolGNN` / `DDIGNNModel` architecture from `ddi_gnn_config.json` and loads the saved weights. Given two drug SMILES strings, predicts the top-k most probable interaction types from the 5-class GNN model.

#### Module C — Food Behavior
Loads the food classifier, `MultiLabelBinarizer`, food lookup table, and BioBERT. For any drug, returns predicted food behavior categories either via the lookup table (exact match on known drugs, more reliable) or via the model (for unknown drugs, from SMILES structure alone).

#### Interactive UI
An `ipywidgets`-based interface with two modes:
- **Drug-Drug Interaction mode:** enter two drugs (by name or SMILES) → get side-by-side RF and GNN predictions for the DDI type plus food behavior for each drug. Drug names are resolved to SMILES automatically via PubChem.
- **Single Drug mode:** enter one drug → food behavior only.

#### Model Comparison Section
Draws a shared held-out sample of DDI pairs and runs both models (RF vs. GNN) head-to-head. Produces: accuracy comparison, per-class F1 bar charts for both models, and a confusion matrix.

#### Programmatic API
Runs the logic of the interactive UI as hardcode for Aspirin and Warfarin. Can be filled in with other names for programattic usage of the tool or is helpful as a debugging/sanity check tool to make sure all parts of the Interactive UI pipeline are working correctly. 

---

## Datasets

| Dataset | Source | Size | Task |
|---|---|---|---|
| DrugBank DDI | [PyTDC](https://tdcommons.ai/) (`tdc.multi_pred.DDI`) | 191,808 pairs, 1,706 drugs | Multi-class DDI type prediction |
| Drug-Food Interactions | [Kaggle](https://www.kaggle.com/datasets/shayanhusain/drug-food-interactions-dataset) | 1,423 drugs | Multi-label food behavior prediction |

---

## Dependencies

```bash
pip install 'numpy<2' rdkit scikit-learn PyTDC matplotlib seaborn
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install transformers kagglehub pubchempy ipywidgets
pip install scikit-learn==1.2.2  # required for model compatibility in Notebook 3
```

All notebooks are designed to run on **Google Colab** with GPU acceleration and Google Drive mounted at `/content/drive/MyDrive/drug_interaction_models`.
