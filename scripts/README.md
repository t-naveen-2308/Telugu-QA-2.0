# Scripts Reference

Scripts organized by workflow stage.

## 📁 Folder Structure

```
scripts/
├── data_collection/   # 1. Gather raw data
├── qa_generation/     # 2. Generate QA pairs
├── augmentation/      # 3. Augment training data
├── data_prep/         # 4. Convert & validate data
├── export/            # 5. Package for Colab training
├── evaluation/        # 6. Evaluate trained models
└── testing/           # 7. Test specific features
```

---

## 1. data_collection/ - Gather Raw Data

| Script | Purpose |
|--------|---------|
| `scrape_news.py` | Scrape Telugu news articles |
| `scrape_government.py` | Scrape government documents |
| `collect_literature.py` | Collect Telugu literature |
| `scrape_real_data.py` | Scrape from multiple sources |
| `scrape_and_mix.py` | Combine real + synthetic data |
| `generate_scaled_data.py` | Scale up dataset size |
| `download_kaggle.py` | Download from Kaggle |

## 2. qa_generation/ - Generate QA Pairs

| Script | Purpose |
|--------|---------|
| `generate_qa_pairs.py` | Generate question-answer pairs from text |

## 3. augmentation/ - Augment Training Data

| Script | Purpose |
|--------|---------|
| `augment_qa.py` | Augment QA pairs (paraphrase, synonym) |

## 4. data_prep/ - Convert & Validate Data

| Script | Purpose |
|--------|---------|
| `convert_to_squad.py` | Convert data to SQuAD format |
| `verify_squad.py` | Validate SQuAD JSON format |
| `analyze_tequad_format.py` | Analyze TeQuAD structure |
| `audit_domain_data.py` | Audit domain data quality |
| `_verify_mix.py` | Verify mixed data ratios |

**Common Usage:**
```bash
# Convert raw data to SQuAD format
python scripts/data_prep/convert_to_squad.py input.json output.json

# Validate the converted file
python scripts/data_prep/verify_squad.py output.json
```

## 5. export/ - Package for Colab Training

| Script | Purpose |
|--------|---------|
| `export_for_colab.py` | Create colab-upload.zip for base training |
| `export_domain_for_colab.py` | Create domain data zip for LoRA training |

**Common Usage:**
```bash
# Export base training data
python scripts/export/export_for_colab.py

# Export domain data
python scripts/export/export_domain_for_colab.py --domain government
```

## 6. evaluation/ - Evaluate Trained Models

| Script | Purpose |
|--------|---------|
| `evaluate_model.py` | Evaluate single model (EM, F1) |
| `evaluate_domain_models.py` | Evaluate all domain models |
| `evaluate_morphology.py` | Evaluate morphology processing |
| `run_evaluation.py` | Full evaluation suite |

**Common Usage:**
```bash
# Evaluate MuRIL model
python scripts/evaluation/evaluate_model.py --model muril

# Run full evaluation
python scripts/evaluation/run_evaluation.py
```

## 7. testing/ - Test Specific Features

| Script | Purpose |
|--------|---------|
| `test_morphology.py` | Test suffix removal, question rules |
| `test_qa_coref.py` | Test coreference resolution |
| `test_vijayawada.py` | Test location resolution |

**Common Usage:**
```bash
# Test morphology processing
python scripts/testing/test_morphology.py
```

---

## Workflow Example

```bash
# 1. Collect data
python scripts/data_collection/scrape_news.py

# 2. Generate QA pairs
python scripts/qa_generation/generate_qa_pairs.py

# 3. Convert to SQuAD format
python scripts/data_prep/convert_to_squad.py raw.json data/domain/news/train.json

# 4. Validate
python scripts/data_prep/verify_squad.py data/domain/news/train.json

# 5. Export for training
python scripts/export/export_domain_for_colab.py --domain news

# 6. Train in Colab (notebooks/04_domain_lora_training.ipynb)

# 7. Evaluate
python scripts/evaluation/evaluate_domain_models.py
```
