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

Each subgraph uses an independent multi-head attention:

$$
e_{ij}^{(k)} = \text{LeakyReLU}(\mathbf{a}_k^T [\mathbf{W}_k \mathbf{h}_i \, \| \, \mathbf{W}_k \mathbf{h}_j]),
\quad
\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}(i)} \exp(e_{il}^{(k)})}.
$$

Updated node embeddings:

$$
\mathbf{h}_i' = \big\|_{k=1}^K \sigma\!\Big(\!\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}_k \mathbf{h}_j\Big).
$$

Cell–cell interactions are encoded by a dedicated CellChat GAT layer, which outputs an attention-based weight to modulate single-cell expression features multiplicatively.

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

$$
\mathcal{L}_{\text{Cox}} = -\sum_{i} \delta_i \Big(\eta_i - \log \sum_{j \in R_i} e^{\eta_j}\Big).
$$

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





---

## 🧠 4. Graph Attention Mechanism

Each subgraph uses an independent multi-head attention mechanism:

```
e_ij^(k) = LeakyReLU( a_k^T [ W_k * h_i || W_k * h_j ] )

α_ij^(k) = exp( e_ij^(k) ) / sum_{l ∈ N(i)} exp( e_il^(k) )
```

The updated node embedding for node *i* is aggregated across all heads *k = 1 ... K*:

```
h'_i = CONCAT_{k=1}^K σ ( sum_{j ∈ N(i)} α_ij^(k) * W_k * h_j )
```

where `σ` denotes a non-linear activation (e.g., ELU), and `CONCAT` represents the concatenation of outputs from each attention head.

To encode cell–cell communication, a dedicated **CellChat-GAT layer** processes the cell–cell graph and produces a learned contextual weight that modulates the single-cell expression matrix multiplicatively.

---

如果需要更 LaTeX 风格，又想保持在 GitHub 上不报错，你也可以用纯行内公式（推荐这样写）：

> **Mathematical Form:**
>
> * Raw attention coefficient for head *k*:
>   `$ e_{ij}^{(k)} = \text{LeakyReLU}(\mathbf{a}_k^\top [\, \mathbf{W}_k \mathbf{h}_i \, || \, \mathbf{W}_k \mathbf{h}_j ]) $`
> * Normalization:
>   `$ \alpha_{ij}^{(k)} = \exp(e_{ij}^{(k)}) / \sum_{l \in \mathcal{N}(i)} \exp(e_{il}^{(k)}) $`
> * Updated embedding:
>   `$ \mathbf{h}'_i = \big\|_{k=1}^K \sigma \big( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}_k \mathbf{h}_j \big) $`

---
