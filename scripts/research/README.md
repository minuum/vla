# Research Scripts

This directory contains scripts for analyzing and validating the Mobile-VLA model and dataset.

## Context Vector Analysis

`analyze_context_vectors.py` validates whether the pre-trained RoboVLMs model (designed for 7DOF manipulators) extracts meaningful context vectors from 2DOF mobile robot data.

### Prerequisites

- **GPU**: A machine with CUDA support is highly recommended (required for efficient inference).
- **RoboVLMs**: The `RoboVLMs` repository must be present in the project root.
- **Dataset**: Mobile-VLA H5 dataset files.

### Usage

Run the script from the project root:

```bash
python3 scripts/research/analyze_context_vectors.py \
    --data_dir path/to/mobile_vla_dataset \
    --output_dir results/context_analysis \
    --max_episodes 50
```

### Output

The script will generate:
- `context_vector_analysis.png`: A plot showing PCA and t-SNE visualizations of the context vectors, colored by action category.
- `features.npz`: Raw feature vectors and labels for further analysis.

### Interpretation

- **Clustering**: If the points cluster well by action category (e.g., all 'Forward' points are together), it indicates the pre-trained model is extracting meaningful features for the mobile robot task.
- **Overlap**: Significant overlap between distinct actions (e.g., 'Left' vs 'Right') suggests the domain gap might be too large, and fine-tuning or adaptation layers are necessary.
