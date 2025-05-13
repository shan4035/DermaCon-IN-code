
# `Metadata_schema.md`
**Schema for `Skin_Metadata.csv`**

| Column Name                       |    Type         |    Description                                                                                  |    Allowed Values / Notes                                                                                                                                         |
|-----------------------------------|--------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Image_name`                      |    String       |    Unique filename or image identifier.                                                        |     Image names (can have different naming conventions). Example: `"IMG_0001.jpg"`                                                                                                                                       |
| `Subject_ID`                      |    String       |    Unique subject/patient identifier.                                                          |     Example: `"SUB00001"`                                                                                                    |
| `Quality`                         |    String       |    Diagnostic image quality assessment.                                                        |     `"Yes, the quality is sufficient for diagnosis"`, `"No, the quality is insufficient for diagnosis"`, `"Image looks normal"`                                   |
| `Sex`                             |    String       |    Biological sex of the subject.                                                              |     `"Male"`, `"Female"`                                                                                                                           |
| `Age`                             |    String       |    Age group of the subject in years.                                                          |     `"0 - 10"`, `"10 - 20"`, ..., `"80 - 100"`                                                                                                                     |
| `Gradability`                     |    String       |    Whether the image is gradable for diagnosis.                                                |     `"Yes"` or `"No"`                                                                                                              |
| `Fitzpatrick`                     |    String       |    Fitzpatrick Skin Type.                                                                      |     `"FST 1"` to `"FST 6"`                                                                                                                                         |
| `Monk_skin_tone`                  |    String       |    Monk Skin Tone scale value.                                                                 |     `"MST 1"` to `"MST 10"`                                                                                                                     |
| `Body_part`                       |    String       |    Anatomical region captured in the image.                                                    |     50 body parts, e.g., `"Back"`, `"Armpits (Axillary Region)"`, `"Scalp"`, `"Face"`, `"Genital Area (Pubic Region)"`, etc.                                     |
| `Descriptors`                        |    String       |    Descriptive visual/morphologic concepts associated with the lesion.                      |     47 lesion descriptors, e.g., `"Abscess"`, `"Bulla"`, `"Plaque"`, `"Crust"`, `"Vesicle"`; may contain multiple values separated by semicolons                                          |
| `Main_class` |    String     |    Broad etiological disease class based on Rook's taxonomy.                                   |     8 Main class labels: `"Infectious Disorders"`, `"Inflammatory Disorders"`, `"Neoplasms and tumors"`, `"Other skin disorders"`, `"No Definite Diagnosis"`                          |
| `Sub_class`                       |    String       |    Specific disease subtype within the main class.                                             |     19 Sub class labels, e.g., `"Hair Disorders"`, `"Keratinisation Disorders"`, `"Infectious skin conditions - Fungal"`, `"Bacterial"`, `"Parasitic"`                                  |
| `Disease_label`                   |    String       |    Disease label having include long-tail classes.                                             |     245 Disease label mapping to dermatological ontology                                                               |
| `Confidence`                      |    Int          |    Diagnostic confidence rating by annotator.                                                  |     Integer from `1` to `5`                                                                                                                                         |
---

### Additional Notes:

- **Missing values** are represented as empty strings or `"N/A"`.
- Multiple `Concepts` or hierarchical annotations apply to a single image.
- This metadata supports **multi-concept** (skin lesion descriptors and body parts), **hierarchical-labels**, and **fine-grained** dermatological annotations to reflect real-world clinical heterogeneity.


**Schema for `train_split.csv` and `test_split.csv`**

- All columns from `Skin_Metadata.csv` are retained as it is except for `Body_part` and `Descriptors`.
- `Body_part` and `Descriptors` are converted into binary columns with 1 indicating presence and 0 indicating absence.
- A stratified split of 80:20 was performed on `Skin_Metadata.csv` based on `Sub_class`
