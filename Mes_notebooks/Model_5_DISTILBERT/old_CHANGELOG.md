# Changelog – Modèle 5 : DistilBERT

## [0.1.0] – 2025-11-20
### Données
- Source : Sentiment140 (`training.1600000.processed.noemoticon.csv`, encodage latin-1).
- Sous-échantillonnage stratifié : 8 000 tweets négatifs (0) + 8 000 positifs (4 → 1), total 16 000 (≈10 % du corpus).
- Colonnes conservées : `sentiment`, `tweet` ; remappage des labels vers {0, 1}.

### Prétraitement
- Nettoyage minimal : trim + réduction des espacements multiples via regex `\s+` (pas de lowercasing/stopwords).
- Split stratifié 80/10/10 (train/val/test) réalisé avant normalisation pour éviter les fuites.
- Conversion en `datasets.Dataset` Hugging Face pour chaque split (`text`, `label`).

### Tokenisation
- Tokenizer `distilbert-base-uncased` (fast) avec troncature `max_length=128` et padding dynamique au chargement (`DataCollatorWithPadding`).
- Batch size 16 pour train/val/test.

### Entraînement
- Fine-tuning `AutoModelForSequenceClassification` (2 labels) via `Trainer`.
- Hyperparamètres : lr 2e-5, 3 epochs, weight decay 0,01, warmup ratio 0,1, fp16, seed 42, 4 workers, pin_memory.
- Stratégie : évaluation + sauvegarde à chaque epoch, `load_best_model_at_end`, patience d’early stopping = 2 (seuil 0), métrique de sélection = f1.
- Métriques calculées : accuracy, precision, recall, f1.

### Suivi et évaluation
- Évaluation validation puis test avec `trainer.evaluate`/`predict`.
- Suivi MLflow : courbe de perte, matrice de confusion, ROC/AUC, métriques val/test (loss, accuracy, precision, recall, f1, AUC).
- Exemple d’inférence conservé (`INFERENCE_EXAMPLE`) pour vérification manuelle.

### Packaging et artefacts
- Export complet dans `distilbert_model_package` : modèle HF (`hf_model`), tokenizer, `model.pkl`, exemples d’inférence (JSON), pipeline de prétraitement (normalisation d’espaces).
- Fichiers d’environnement générés (`requirements.txt`, `conda.yaml`, `python_env.yaml`) + signature/artefacts consignés dans MLflow (`model_package`, modèle PyTorch loggé).

## [0.2.0] – 2025-11-20

### Données
- Source : Sentiment140 (`training.1600000.processed.noemoticon.csv`, encodage latin-1).
- Changement de la taille de l'échantillon de 16000 (8000*2) à 20000 (10000*2)

### Entraînement
- Fine-tuning `AutoModelForSequenceClassification` (2 labels) via `Trainer`.
- Hyperparamètres : 3 epochs -> 20 epochs.