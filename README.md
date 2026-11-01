# Traffic Sign Classification with Deep Learning

Academic project exploring recognition of 43 traffic sign classes from the German Traffic Sign Recognition Benchmark (GTSRB).

## Overview

The notebook includes:
- A fully connected MLP baseline
- A LeNet-5-inspired CNN
- Experiments with dropout, batch normalization, and image augmentation
- Zero-shot CLIP evaluation
- Learning curves, confusion matrices, and misclassification analysis

## Technologies

Python, PyTorch, torchvision, NumPy, pandas, Matplotlib, scikit-learn, and Hugging Face Transformers.

## Dataset

GTSRB contains 43 traffic sign categories, including speed limits, warning signs, and directional signs.

The notebook downloads the official training and test splits through torchvision. It creates a validation subset from 15% of the official training split using seed 42.

Supervised models use 32 × 32 RGB images normalized with ImageNet statistics.

Dataset files are not included in this repository.

## Saved Results

Values below come from the notebook's saved outputs.

| Model / Experiment | Test Accuracy | Macro F1 |
|---|---:|---:|
| MLP baseline | 78.69% | 0.7245 |
| Reference CNN | 91.36% | 0.8764 |
| EXP0: dropout 0.3, no BatchNorm | 92.22% | 0.8878 |
| EXP1: full training loader, dropout 0.3 | 92.92% | 0.9061 |
| EXP2: dropout 0.2 | 92.39% | 0.8848 |
| EXP3: dropout 0.5 | 91.17% | 0.8837 |
| EXP4: dropout 0.3 with BatchNorm | 92.51% | 0.8869 |
| Zero-shot CLIP | 30.45% | Not recorded |

EXP1 recorded the highest test accuracy among the saved CNN experiments. The evaluation limitations below apply to these results.

## Training Configuration

- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10 per supervised run
- CNN batch size: 64
- MLP batch size: 128
- CNN checkpoint selection: highest validation accuracy

## Running the Notebook

1. Open `traffic_sign_classification.ipynb` in Google Colab or Jupyter.
2. Install dependencies in a notebook cell:

```python
%pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm transformers
```

3. Run the notebook cells sequentially.

Internet access is required to download GTSRB and pretrained CLIP weights. A GPU can accelerate model computation.

The notebook uses `open_clip` if installed; otherwise, it tries Hugging Face Transformers. The saved CLIP evaluation used Transformers.

Saved tables and plots can be viewed directly on GitHub without running the notebook.

## Evaluation Limitations

- EXP1 trains using the full official training split, including images assigned to validation. Its validation-based checkpoint selection is therefore not independent.
- Other CNN runs use an augmented training dataset despite their original non-augmentation labels. These experiments do not isolate augmentation's effect.
- The winning experiment is selected using test accuracy, which limits interpretation of its test score as an independent final estimate.
- The random image-level validation split does not enforce separation of related image sequences.
- Subset confusion plots omit predictions outside the selected classes. Their saved macro F1 calculation does not explicitly restrict labels to the intended subset.
- Some displayed error-example images are not denormalized, affecting their colors.

These saved results should be interpreted as exploratory academic experiments.

## Future Improvements

- Separate augmented and non-augmented training loaders correctly.
- Exclude validation images from every training loader.
- Select experiment configurations using independent validation performance.
- Evaluate the selected configuration once on the held-out test set.
- Correct subset metrics and image display normalization.
- Repeat experiments across multiple random seeds.

## Repository Contents

- `traffic_sign_classification.ipynb`: implementation, saved metrics, and visualizations
- `README.md`: project documentation

This project is an academic classification study, not a deployed driving-safety system.
