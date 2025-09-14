# Interactive Residue Predictor

A deep learning model for predicting protein-peptide interaction residues using ESM2 and Cross-Attention mechanisms.

## Overview

This project implements a neural network that predicts which residues in a protein-peptide complex are involved in interactions. The model uses ESM2 (Evolutionary Scale Modeling) for sequence encoding and Cross-Attention to capture interactions between protein and peptide sequences.

## Model Architecture

- **Encoders**: ESM2-t36-3B-UR50D (2560-dimensional embeddings)
- **Cross-Attention**: 16 heads, 160-dimensional heads
- **MLP Networks**: 5-layer ResNet-style architecture with skip connections
- **Output**: Binary classification for each residue (interaction probability)

## Requirements

```bash
pip install torch transformers fair-esm pandas numpy scikit-learn matplotlib tqdm
```

## Data Format

Input CSV files should contain the following columns:
- `pair_key`: Unique identifier for the protein-peptide pair
- `receptor_sequence`: Protein sequence
- `peptide_sequence`: Peptide sequence
- `receptor_vector`: Interaction labels for protein residues (comma-separated 0/1)
- `peptide_vector`: Interaction labels for peptide residues (comma-separated 0/1)

## Usage

### Training

```bash
cd src
python train_model.py
```

Training parameters:
- Epochs: 3
- Batch size: 20
- Learning rate: 1e-4
- Device: CUDA (GPU recommended)

### Inference

```bash
cd src
python inference.py \
    --test_data_path ../data/test_data.csv \
    --model_path ../pth/best_model.pth \
    --output_dir ../test_result \
    --output_type probability
```

### Evaluation and Visualization

```bash
# Generate ROC curves
python util/plot_ROC.py \
    --true_csv data/true_labels.csv \
    --pred_csv result/csv/all_predictions.csv \
    --output result/plot/ROC_curve.png

# Sort predictions by confidence
python util/sort_predicted_prob_indices.py \
    --input result/csv/all_predictions.csv \
    --output result/csv/sorted_predictions.csv
```

## Project Structure

```
├── src/
│   ├── Interactive_residues_predictor_main.py  # Main model definition
│   ├── train_model.py                          # Training script
│   ├── inference.py                            # Inference script
│   ├── sequence_data_encoders.py               # Sequence encoders
│   └── util_for_model_construct.py             # Utility functions
├── data/                                       # Data directory
├── result/                                     # Results directory
├── util/                                       # Evaluation tools
└── test_result/                                # Test results
```

## Key Features

- **Bidirectional Prediction**: Simultaneously predicts interaction residues for both protein and peptide
- **Cross-Attention Mechanism**: Models protein-peptide interactions through attention weights
- **Pre-trained Models**: Leverages ESM2 pre-trained weights for sequence representation
- **Comprehensive Evaluation**: Provides multiple metrics (AUC, F1-score) and visualization tools

## Output Files

- **Best Model**: `pth/best_model.pth`
- **Predictions**: `test_result/all_predictions.csv`
- **Metrics**: `result/csv/AUC_F1_test.csv`
- **Visualizations**: `result/plot/ROC_curve_test.png`

## Evaluation Metrics

- AUC (Area Under Curve)
- F1-Score
- ROC Curves
- Per-residue interaction probabilities

## Example

The model takes protein and peptide sequences as input and outputs interaction probabilities for each residue:

```python
# Input sequences
protein_seq = "GSHSLRYFYTAVSRPGLGEPRFIAVGYVDDTEFVRFDSDAENPRMEPRARWMEREGPEYWEQQTRIAKEWEQIYRVDLRTLRGYYNQSEGGSHTIQEMYGCDVGSDGSLLRGYRQDAYDGRDYIALNEDLKTWTAADFAAQITRNKWERARYAERLRAYLEGTCVEWLSRYLELGKETLLRSDPPEAHVTLHPRPEGDVTLRCWALGFYPADITLTWQLNGEDLTQDMELVETRPAGDGTFQKWASVVVPLGKEQNYTCRVEHEGLPKPLSQRWE"
peptide_seq = "ILFPSSERLISNR"

# Output: Interaction probabilities for each residue
# protein_probs: [0.05, 0.02, 0.03, ..., 0.01]  # Length = protein length
# peptide_probs: [0.85, 0.88, 0.92, ..., 0.94]  # Length = peptide length
```

## License

This project is for research purposes. Please cite appropriately if used in academic work.
