# Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer

## 1. Project Overview

This repository hosts the full reproducible pipeline for our study *“Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer (TNBC)”*. The framework integrates **single-cell transcriptomics**, **chromatin accessibility**, **cell–cell communication**, **radiomics**, and **network interactions** into a unified **multi-layer graph attention network (GAT)**.

The goal is to resolve TNBC’s molecular heterogeneity and discover high-confidence predictive gene modules with both **interpretability** and **prognostic relevance**.

---

## 2. Datasets and Preprocessing

**Data sources:**

* **GSE199219:** scRNA-seq + CITE-seq for cell-type-specific expression and surface markers.
* **GSE168026:** scATAC-seq for TF motif inference and chromatin state.
* **TCGA-BRCA:** Bulk RNA-seq for survival modeling.
* **TCIA Radiomics:** Handcrafted image features from segmented MRI/CT scans.
* **STRING, HGD, CellPhoneDB:** Protein-protein interactions, homologous genes, ligand–receptor edges.

**Key preprocessing steps:**

* *scRNA-seq*: Cells filtered if genes detected <200 or >7,500; >20% mitochondrial content excluded; normalized by log transform; scaled with variable genes; cell-cycle effect regressed out.
* *scATAC-seq*: Processed by **CellRanger-ATAC**, integrated via *Seurat* + *Signac*; QC by unique fragments, % reads in peaks, TSS enrichment, nucleosomal signal.
* *Radiomics*: 35 handcrafted features projected into the same latent space as the transcriptome by **Canonical Correlation Analysis (CCA)**.

---

## 3. Multi-Layer Heterogeneous Graph Definition

The final integrated graph is defined as:

```
G = (V, E)
  where
    V = {Genes, Cells, Radiomic Features},
    E = {E_PPI, E_Homolog, E_TF, E_CellChat}.
```

* **Nodes:** Genes initialized with PCA scores; radiomics with CCA projections; cells with binary marker genes + cell-cell interaction attributes.
* **Edges:** Include STRING PPI, homologous links (HGD), Pando TF-binding motifs, CellPhoneDB ligand–receptor pairs.

---

## 4. Graph Attention Mechanism

Each subgraph uses an independent multi-head attention:

```
e_ij^(k) = LeakyReLU(a_k^T [W_k h_i || W_k h_j]),
alpha_ij^(k) = exp(e_ij^(k)) / sum_{l in N(i)} exp(e_il^(k)).

h_i' = concat_{k=1}^K sigma( sum_{j in N(i)} alpha_ij^(k) W_k h_j )
```

Cell–cell interactions are encoded by a dedicated CellChat GAT layer, which outputs an attention-based weight to modulate single-cell expression features multiplicatively.

---

## 5. Prognostic MLP with Variational Dropout

Final embeddings pass through:

```
MLP: [256 -> 128 -> 64 -> 1]
```

Dropout is parameterized via the Concrete Dropout trick:

```
~z = Sigmoid( (1/tau) [ log(p) - log(1-p) + log(u) - log(1-u) ] ), u ~ U(0,1)

h~ = h * (1 - ~z)
```

Optimized by partial likelihood:

```
L_Cox = -sum_i delta_i ( eta_i - log sum_{j in R_i} exp(eta_j) ).
```

---

## 6. Radiogenomic Decoder

A deep decoder reconstructs gene expression from radiomics:

```
y^ = f_theta(x),   L_MSE = || y^ - y ||^2.
```

This maps imaging-derived phenotypes to transcriptomic space, checked by rank correlations.

---

## 7. Training, Validation & Reproducibility

* Optimizer: Adam, LR 0.001–0.005.
* 5-fold CV for stability.
* Evaluation: AUC-ROC, AUPR, F1-score, Concordance Index (C-index).
* Random seeds fixed for reproducibility: `seed = 4709`.

---

## 8. Run This Project

```bash
# Clone the repo
git clone https://github.com/jjjjhengxin/Triple_Negative_Breast_Cancer_Multi_Omics.git
cd Triple_Negative_Breast_Cancer_Multi_Omics

# Install dependencies
pip install -r requirements.txt

# Train GAT + MLP
python train.py
```

Outputs: risk predictions, gene importance scores, survival plots, and full logs under `result/`.

---

## Citation

If you find this useful, please cite our manuscript and refer to the SI for full theoretical derivations.
