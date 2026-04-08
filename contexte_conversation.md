# Contexte complet du projet — Prédiction de notes d'hôtels

## Objectif
Prédire la note (1★ à 5★) d'un avis d'hôtel à partir du texte du commentaire.
Tâche : **classification 5 classes** (pas de régression, pas de regroupement en 3 classes — imposé par le prof).
Objectif de performance : **80% d'accuracy**.

---

## Données

| Fichier | Lignes | Description |
|---|---|---|
| `train_hotel_reviews.csv` | 18 491 | Entraînement |
| `valid_hotel_reviews.csv` | 1 000 | Validation |
| `test_hotel_reviews.csv` | 1 000 | Test |

Colonnes : `Review` (texte), `Rating` (note 1 à 5).

Distribution train (très déséquilibrée) :
- 1★ : 1 310 (7%)
- 2★ : 1 630 (9%)
- 3★ : 1 982 (11%)
- 4★ : 5 465 (30%)
- 5★ : 8 104 (44%)

---

## Structure du projet

```
Prédiction de notes à partir de commentaires sur des hôtels/
├── train_hotel_reviews.csv
├── valid_hotel_reviews.csv
├── test_hotel_reviews.csv
├── README.md
├── requirements.txt
├── rapport.tex                          ← rapport LaTeX (à mettre à jour)
├── contexte_conversation.md             ← ce fichier
├── venv/                                ← environnement Python local
├── TF-IDF/
│   └── TF-IDF.ipynb                    ← FAIT, résultats obtenus
├── Sentence Transformers/
│   └── Sentence_Transformers.ipynb     ← FAIT, résultats obtenus
├── Fine-tuning BERT/
│   └── BERT.ipynb                      ← version locale (NaN loss sur MPS, inutilisable)
├── Fine-tuning BERT Colab/
│   ├── BERT.ipynb                      ← roberta-base, 3 époques, FAIT
│   ├── best_roberta.pt
│   └── predictions_roberta.csv
├── Fine-tuning BERT Colab roberta-large/
│   ├── BERT_roberta_large.ipynb        ← roberta-large, 5ep, class weights, FAIT
│   ├── best_roberta_large.pt
│   └── predictions_roberta_large.csv
└── Fine-tuning BERT nlptown/
    └── BERT.ipynb                      ← EN COURS (erreur 404 nlptown)
```

---

## Résultats obtenus

### Tableau complet

| Modèle | Accuracy (test) | F1 macro (test) | Statut |
|---|---|---|---|
| TF-IDF + LogReg | 61.3% | 0.493 | FAIT |
| TF-IDF + LinearSVC | 60.4% | — | FAIT |
| Sentence Transformers + LogReg | 63.5% | 0.566 | FAIT |
| RoBERTa-base fine-tuné (3 ep) | **70.4%** | **0.636** | FAIT |
| RoBERTa-large (5ep + class weights) | 69.6% | 0.650 | FAIT |
| nlptown fine-tuné | — | — | ERREUR 404 |

### Détail RoBERTa-base (meilleur modèle jusqu'ici)

| Note | Précision | Rappel | F1 |
|---|---|---|---|
| 1★ | 0.69 | 0.75 | 0.72 |
| 2★ | 0.56 | 0.60 | 0.58 |
| 3★ | 0.61 | 0.35 | 0.44 ← point faible |
| 4★ | 0.60 | 0.60 | 0.60 |
| 5★ | 0.81 | 0.86 | 0.83 |

Erreurs : 296/1000 — 94% à écart ≤1 étoile (erreurs légères).

---

## Décisions techniques prises

### Prétraitement
- **TF-IDF** : nettoyage agressif (lower, stopwords NLTK, ponctuation, chiffres)
- **Sentence Transformers** : texte brut (strip uniquement)
- **BERT/RoBERTa** : texte brut, tokenizer propre au modèle
- Labels : toujours `Rating - 1` (0 à 4) pour PyTorch

### Modélisation
- Classification (pas régression) — décision imposée
- 5 classes maintenues (pas de regroupement — imposé par prof)
- Class weights calculés avec `sklearn.utils.class_weight.compute_class_weight`
- Early stopping patience=2

### Hyperparamètres RoBERTa-base (meilleure config)
- `MODEL_NAME = 'roberta-base'`
- `MAX_LEN = 256`
- `BATCH_SIZE = 32`
- `LR = 2e-5`
- `EPOCHS = 3` (overfitting dès époque 3)
- `AdamW + scheduler linéaire avec 10% warmup`

### Problèmes rencontrés
1. **NaN loss sur MPS (Mac M1)** : bug PyTorch/MPS avec RoBERTa → migré sur Colab
2. **roberta-large** : gain marginal (+0.015 F1) car max_length réduit à 128 pour tenir en VRAM
3. **nlptown 401 puis 404** : modèle nécessite auth HF + possiblement renommé/déplacé
4. **Overfitting RoBERTa** : F1 valid baisse dès époque 3 (0.6316 → 0.6295)

---

## Environnement

- Python 3.11 (venv local)
- PyTorch 2.10.0
- Transformers 5.3.0
- scikit-learn 1.5.1
- sentence-transformers installé
- Colab : GPU T4 (15.6 Go VRAM), CUDA

---

## Problème en cours à résoudre

**nlptown/bert-base-multilingual-cased-sentiment → erreur 404**

Ce modèle était pré-entraîné sur des avis hôtels/restaurants 1-5★ (TripAdvisor, Yelp, Amazon).
Erreur 404 = modèle introuvable malgré authentification HF.
→ Causes possibles : modèle renommé, supprimé, ou gated nécessitant acceptation manuelle des CGU.
→ Solution à trouver : alternative publique équivalente ou acceptation CGU sur HF.

---

## Pistes restantes pour atteindre 80%

1. **Trouver alternative à nlptown** (modèle pré-entraîné sur reviews 1-5★ accessible)
2. **Augmenter données classes rares** (2★, 3★) par data augmentation
3. **Ensemble** : combiner prédictions RoBERTa-base + autre modèle

---

## Token HuggingFace

⚠️ Un token HF a été partagé dans la conversation — **doit être régénéré** sur huggingface.co/settings/tokens.
