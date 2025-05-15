# DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research

The repository in Harvard Dataverse contains the code, models, and metadata used for training and evaluating multi-concept dermatology models on the DermaCon-IN dataset.

---
```
## 📁 Folder Structure

DermaCon-IN/
│
├── checkpoints/ # Pretrained model checkpoints
│ ├── CBM_MC_best_model.pth # Best Concept Bottleneck Model checkpoint
│ ├── CBM_SC_MC_type1.pth # CBM model with Sub-Class Main-Class (Type 1)
│ ├── CBM_SC_MC_type2.pth # CBM model with Sub-Class Main-Class (Type 2)
│ ├── Swin_MC_best_model.pth # Best Swin Transformer model checkpoints
│
├── DATASET/ # all images in .jpg format
│
├── METADATA/ # Metadata used for training/analysis
│ ├── Skin_Metadata.csv # Core metadata per image (age, sex, labels, etc.)
│ ├── test_split.csv # Test split 
│ ├── train_split.csv # Training split 
│ ├── Metadata_schema.md # Description of metadata columns and schema
│
├── Infer_CBM_MC.ipynb # Inference notebook for CBM Main-Class model
├── Infer_Swin_MC.ipynb # Inference notebook for Swin Main-Class model
│
├── README.md # This file
│
├── train_cbm_mc.py # Training script for CBM Main-Class model
├── train_CBM_SC_MC_type1.py # Training script for CBM with SC-MC Type 1
├── train_CBM_SC_MC_type2.py # Training script for CBM with SC-MC Type 2
├── train_swin_mc.py # Training script for Swin Transformer Main-Class model
```


---

## 🧠 Model Variants

- **CBM_MC**: Concept Bottleneck Model trained on multi-concept annotations for Main-Class classification.
- **CBM_SC_MC_type1/2**: Variants of Sub-Class Main-Class classification using multi-concept annotations (end-to-end joint training).
- **Swin_MC**: Vision Transformer-based Swin model trained for Main-Class classification.

---

## 📦 Metadata

- `Skin_Metadata.csv`: Per-image metadata including demographics and quality flags.
- `train/test_split.csv`: Official splits used for training and evaluation.
- `Metadata_schema.md`: Documentation of metadata column definitions.

---

## 🚀 Training Scripts

Each training script corresponds to a specific model variant. Modify the script arguments or configs inside the script as needed.

---

## 📒 Inference

Use the `Infer_CBM_MC.ipynb` and `Infer_Swin_MC.ipynb` notebooks to run inference and analyze prediction results.
