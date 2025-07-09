---

# Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer

## 📌 1. Project Overview

This repository hosts the full reproducible pipeline for our study *“Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer (TNBC)”*. The framework integrates **single-cell transcriptomics**, **chromatin accessibility**, **cell–cell communication**, **radiomics**, and **network interactions** into a unified **multi-layer graph attention network (GAT)**.

The goal is to resolve TNBC’s molecular heterogeneity and discover high-confidence predictive gene modules with both **interpretability** and **prognostic relevance**.

---

## 📊 2. Datasets and Preprocessing

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

## 🔗 3. Multi-Layer Heterogeneous Graph Definition

The final integrated graph is defined as:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
\quad \text{where} \quad
\mathcal{V} = \{\text{Genes, Cells, Radiomic Features}\}, \quad
\mathcal{E} = \{E_{PPI}, E_{Homolog}, E_{TF}, E_{CellChat}\}.
$$

* **Nodes:** Genes initialized with PCA scores; radiomics with CCA projections; cells with binary marker genes + cell-cell interaction attributes.
* **Edges:** Include STRING PPI, homologous links (HGD), Pando TF-binding motifs, CellPhoneDB ligand–receptor pairs.

---

## 🧠 4. Graph Attention Mechanism

Each subgraph in our multi-layer framework is processed by an independent multi-head attention module. For every node, the model learns how strongly to weight its neighbors when updating its representation. The raw attention scores are calculated using a shared linear transformation and a learnable attention vector, followed by a non-linear activation.

The scores are normalized across each node’s neighborhood so that they sum to one, ensuring that closer or more relevant neighbors contribute more to the updated node feature. Multiple attention heads run in parallel, and their outputs are concatenated to capture diverse aspects of the graph structure.

To model cell–cell communication explicitly, we add a dedicated attention layer for the cell–cell graph derived from CellChat. This layer computes a contextual weight that adjusts single-cell expression data, providing extra biological context during graph message passing.

---

## 🩺 5. Prognostic MLP with Variational Dropout

Final embeddings pass through:

$$
\text{MLP}: [256 \to 128 \to 64 \to 1].
$$

Dropout is parameterized via the Concrete Dropout trick:

$$
\tilde{z} = \sigma\!\Big(\!\frac{1}{\tau} [\log p - \log(1\!-\!p) + \log u - \log(1\!-\!u)]\Big), \quad u \sim U(0,1),
\quad \tilde{\mathbf{h}} = \mathbf{h} \odot (1 - \tilde{z}).
$$

Optimized by partial likelihood:

L\_Cox = - Σ₍ᵢ₎ δᵢ \[ ηᵢ – log Σⱼ∈Rᵢ exp(ηⱼ) ]


---

## 🧬 6. Radiogenomic Decoder

A deep decoder reconstructs gene expression from radiomics:

$$
\hat{\mathbf{y}} = f_{\theta}(\mathbf{x}), \quad
L_{\text{MSE}} = \|\hat{\mathbf{y}} - \mathbf{y}\|_2^2.
$$

This maps imaging-derived phenotypes to transcriptomic space, checked by rank correlations.

---

## ✅ 7. Training, Validation & Reproducibility

* Optimizer: Adam, LR 0.001–0.005.
* 5-fold CV for stability.
* Evaluation: AUC-ROC, AUPR, F1-score, Concordance Index (C-index).
* Random seeds fixed for reproducibility: `seed = 4709`.

---

## 🚀 8. Run This Project

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

## 📑 Citation

If you find this useful, please cite our manuscript and refer to the SI for full theoretical derivations.

---
