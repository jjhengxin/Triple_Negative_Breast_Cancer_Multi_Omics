# Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer

## 1. Project Overview

This repository contains the full implementation of the study *“Graph Attention Network-Based Multi-Omics Data Integration for Precise Key Gene Prediction in Triple-Negative Breast Cancer (TNBC)”*, as described in our manuscript and accompanying supplementary information (SI). Our pipeline systematically integrates single-cell transcriptomics, chromatin accessibility, cell-cell communication, radiomics and protein interaction networks within a unified graph attention network (GAT) architecture.

The objective is to decode the molecular and cellular heterogeneity of TNBC and derive robust predictive signatures with both interpretability and prognostic power.

---

## 2. Dataset Summary and Preprocessing Rationale

The project employs multiple public and curated datasets:

- **scRNA-seq + CITE-seq (GSE199219)** for cell-type specific transcriptional and surface protein profiling.
- **scATAC-seq (GSE168026)** for chromatin accessibility and transcription factor (TF) motif enrichment.
- **Bulk RNA-seq (TCGA-BRCA)** for population-level survival modeling.
- **Radiogenomics (TCGA Radiomics)** for imaging-derived phenotype embedding.
- **STRING, HGD, CellPhoneDB** for PPI, homology and ligand-receptor edges.

Preprocessing follows standard best practices to mitigate technical noise and preserve true biological signals. For scRNA-seq, we filter cells with <200 or >7,500 genes detected and >20% mitochondrial RNA content. Normalization is performed by log-transform and scaling with variable gene selection. scATAC-seq peaks are called via CellRanger-ATAC and integrated with Seurat + Signac, with TSS enrichment, nucleosomal signal and fragment counts as QC metrics.

For radiomics, 35 handcrafted features are extracted post-lesion segmentation. These features are projected into the same latent space as the transcriptomic principal components via Canonical Correlation Analysis (CCA) to bridge the imaging-transcriptomic modality gap.

---

## 3. Multi-Layer Heterogeneous Graph Definition and Construction

The unified multi-omics graph is formally defined as a heterogeneous graph \( \mathcal{G} = (\mathcal{V}, \mathcal{E}) \) where:
- The node set \( \mathcal{V} \) includes gene nodes, cell type nodes and radiomic feature nodes.
- The edge set \( \mathcal{E} \) comprises protein-protein interactions (STRING), homologous gene links (HGD), transcription factor bindings (Pando-inferred) and cell-cell ligand-receptor interactions (CellPhoneDB).

Mathematically, the complete edge index is stored in disjoint subgraphs:
\[
\mathcal{E} = \{ E_{PPI}, E_{Homolog}, E_{ATAC1}, E_{ATAC2}, E_{CellChat} \}.
\]

Node representations are initialized as:
- Genes: top principal components (PCA) of normalized expression profiles.
- Radiomic features: CCA-projected scores.
- Cells: binary indicator vectors for marker genes plus interaction scores.

---

## 4. Cross-Modal Graph Attention Mechanism: Derivation and Implementation

Each modality-specific subgraph is processed by an independent attention layer. For each node \( i \) and neighbor \( j \), the raw attention coefficient for head \( k \) is computed as:

\[
e_{ij}^k = \text{LeakyReLU} \left( \mathbf{a}_k^T [\mathbf{W}_k \mathbf{h}_i || \mathbf{W}_k \mathbf{h}_j] \right),
\]

where \( \mathbf{W}_k \) is the learnable linear transformation for head \( k \), and \( \mathbf{a}_k \) is the learnable attention vector. The attention coefficients are normalized across the neighborhood \( \mathcal{N}(i) \):

\[
\alpha_{ij}^k = \frac{\exp(e_{ij}^k)}{\sum_{l \in \mathcal{N}(i)} \exp(e_{il}^k)}.
\]

Updated node embeddings are then aggregated as:

\[
\mathbf{h}_i' = \Big\|_{k=1}^K \sigma \Big( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}_k \mathbf{h}_j \Big),
\]

where \( K \) is the total number of attention heads and \( \| \) denotes concatenation.

To robustly encode cell-cell communication, a dedicated GAT layer processes the CellChat-derived cell-cell graph, producing a contextual weight that modulates the expression matrix multiplicatively.

---

## 5. Prognostic MLP with Variational Dropout: Theoretical Basis

The final node embeddings are fed into a survival prediction module based on a multi-layer perceptron (MLP). The architecture is defined as:

\[
\text{MLP}: \quad [256 \rightarrow 128 \rightarrow 64 \rightarrow 1],
\]

with ReLU activations. Dropout regularization is implemented via a variational, learnable mask inspired by the Concrete Dropout paradigm. Specifically, the dropout probability \( p \) is parameterized and relaxed via the Gumbel-Softmax trick:

\[
\tilde{z} = \text{Sigmoid} \left( \frac{1}{\tau} \Big( \log p - \log (1 - p) + \log u - \log (1 - u) \Big) \right),
\quad u \sim \text{Uniform}(0,1).
\]

The final output is masked as:

\[
\tilde{\mathbf{h}} = \mathbf{h} \odot (1 - \tilde{z}),
\]

where \( \odot \) denotes element-wise multiplication. The survival risk is predicted by optimizing the Cox proportional hazards partial likelihood:

\[
L_{Cox} = -\sum_{i} \delta_i \Big( \eta_i - \log \sum_{j \in R_i} e^{\eta_j} \Big),
\]

where \( \eta_i \) is the predicted risk score, \( \delta_i \) the event indicator, and \( R_i \) the risk set.

---

## 6. Radiomics-Transcriptome Decoder: Motivation and Derivation

To bridge radiomic features and gene expression, we design a deep decoder network that learns to reconstruct high-dimensional gene profiles from lower-dimensional radiomics embeddings. Formally, given radiomic input \( \mathbf{x} \), the decoder approximates:

\[
\hat{\mathbf{y}} = f_{\theta}(\mathbf{x}),
\quad L_{MSE} = \| \hat{\mathbf{y}} - \mathbf{y} \|_2^2,
\]

where \( f_{\theta} \) denotes the feed-forward layers with ReLU activations and sigmoid output for bounded gene expression estimates. This mapping is validated by evaluating the Spearman or rank-based correlation between reconstructed and original expression matrices, ensuring the biological plausibility of the imaging-transcriptome link.

---

## 7. Training Strategy, Cross-Validation and Reproducibility

The full model is trained with Adam optimizer (learning rate 0.001–0.005) with five-fold cross-validation to guard against overfitting and ensure robust generalization. Performance is evaluated by standard metrics:

- **Classification**: AUC-ROC, AUPR, F1-score.
- **Survival**: Concordance Index (C-index).

All random seeds are fixed to ensure exact reproducibility. Users may adjust the `seed` parameter in `train.py` to reproduce identical splits and results.

---

## 8. How to Run

Clone this repository and install dependencies:
```bash
git clone https://github.com/jjhengxin/Triple_Negative_Breast_Cancer_Multi_Omics.git
cd Triple_Negative_Breast_Cancer_Multi_Omics
pip install -r requirements.txt
