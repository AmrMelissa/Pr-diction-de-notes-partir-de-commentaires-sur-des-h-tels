# Prédiction de notes à partir de commentaires d'hôtels

## Objectif

À partir d'un commentaire textuel laissé par un client sur un hôtel, **prédire la note attribuée** (de 1 à 5 étoiles).

C'est un problème de **traitement automatique du langage naturel (NLP)** où l'on apprend la relation entre le contenu d'un avis et la satisfaction du client.

---

## Données disponibles

| Fichier | Rôle | Taille |
|---|---|---|
| `train_hotel_reviews.csv` | Entraîner le modèle | ~18 500 exemples |
| `valid_hotel_reviews.csv` | Ajuster les hyperparamètres, évaluer en cours d'entraînement | ~1 000 exemples |
| `test_hotel_reviews.csv` | Évaluation finale uniquement (ne pas toucher avant la fin) | ~1 000 exemples |

Chaque fichier contient deux colonnes :
- `Review` : texte du commentaire (en anglais)
- `Rating` : note de 1 à 5 (entier)

---

## Vue d'ensemble du pipeline

```
Données brutes (texte + note)
        ↓
Exploration & analyse
        ↓
Préparation des données
        ↓
Représentation numérique du texte
        ↓
Entraînement du modèle (classification ou régression)
        ↓
Évaluation & comparaison
        ↓
Prédiction sur le jeu de test
```

---

## Catégorie 1 — Exploration des données

**Objectif :** Comprendre la structure et les caractéristiques des données avant toute modélisation.

### Étapes

1. **Charger les CSV** avec `pandas`
2. **Vérifier les valeurs manquantes** : y a-t-il des lignes sans texte ou sans note ?
3. **Distribution des notes** : combien d'exemples par note (1, 2, 3, 4, 5) ? Le jeu est-il équilibré ?
4. **Longueur des commentaires** : distribution du nombre de mots/tokens par avis
5. **Exemples visuels** : lire quelques avis de chaque note pour développer une intuition

### Questions à répondre
- Les notes sont-elles équilibrées ? (si non, il faudra en tenir compte)
- Y a-t-il des outliers (textes très courts ou vides) ?
- Quelle est la longueur maximale / moyenne des textes ?

---

## Catégorie 2 — Préparation des données

**Objectif :** Transformer les données brutes en format exploitable par les modèles.

### 2a. Nettoyage du texte (pour les approches classiques)

Ces étapes sont utiles pour TF-IDF / Bag of Words, **moins nécessaires pour BERT** (qui gère le texte brut) :

1. Mettre le texte en minuscules
2. Supprimer la ponctuation et les caractères spéciaux
3. Supprimer les stopwords (mots très fréquents sans sens : "the", "a", "is"...)
4. Lemmatisation / stemming (réduire les mots à leur racine)

### 2b. Encodage des labels

**Point critique :** selon l'approche choisie, les labels doivent être préparés différemment.

| Approche | Format des labels | Exemple |
|---|---|---|
| **Classification** | Entiers de 0 à N-1 | Note 1 → 0, Note 5 → 4 |
| **Régression** | Valeurs numériques continues | Note 1 → 1.0, Note 5 → 5.0 |

> Pour BERT avec `BertForSequenceClassification`, les labels doivent aller de **0 à 4** (pas de 1 à 5).
> Appliquer : `label = rating - 1`

### 2c. Séparation des données

Les fichiers sont déjà séparés. Règle à respecter :
- Entraîner uniquement sur `train`
- Valider sur `valid` pendant l'entraînement
- Évaluer sur `test` **une seule fois à la fin**

---

## Catégorie 3 — Représentation numérique du texte

**Objectif :** Convertir le texte en vecteurs de nombres que le modèle peut traiter.

### Option A — TF-IDF (baseline simple)

- Chaque mot du vocabulaire devient une dimension
- La valeur = fréquence du mot dans le document, pondérée par sa rareté globale
- Rapide, interprétable, bon point de départ

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_texts)
X_valid = vectorizer.transform(valid_texts)
```

### Option B — BERT (approche avancée)

BERT (Bidirectional Encoder Representations from Transformers) est un modèle pré-entraîné sur des milliards de phrases. Il produit des représentations **contextuelles** : le même mot a une représentation différente selon le contexte.

**Deux façons d'utiliser BERT :**

#### B1. BERT comme extracteur de features (sans fine-tuning)
- BERT reste figé (ses poids ne sont pas modifiés)
- On extrait le vecteur `[CLS]` (vecteur de 768 dimensions représentant la phrase entière)
- Ce vecteur est ensuite passé à un modèle classique (régression logistique, etc.)

```
Texte → BERT (figé) → vecteur [CLS] 768-dim → modèle classique → note
```

#### B2. Fine-tuning BERT (recommandé, meilleure performance)
- On ajoute une tête de classification/régression sur BERT
- On ré-entraîne l'ensemble (BERT + tête) sur nos données
- BERT s'adapte spécifiquement à la tâche de prédiction de notes

```
Texte → BERT (entraîné) + tête → note prédite
```

---

## Catégorie 4 — Modélisation

**Objectif :** Choisir et entraîner le bon modèle selon l'approche retenue.

### Classification vs Régression

| | **Classification** | **Régression** |
|---|---|---|
| Interprétation | Les notes sont 5 catégories distinctes | Les notes sont des valeurs ordonnées |
| Sortie | Probabilité pour chaque note | Valeur continue (à arrondir) |
| Loss | `CrossEntropyLoss` | `MSELoss` ou `MAELoss` |
| Métriques | Accuracy, F1-score | MAE, RMSE |
| Recommandation | **Préféré** pour ce problème | Alternative valable |

### 4a. Baseline : TF-IDF + modèle classique

Objectif : obtenir un score de référence rapide avant BERT.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Modèles à tester :
- Régression Logistique
- SVM linéaire (`LinearSVC`)
- Random Forest

### 4b. Fine-tuning BERT avec HuggingFace

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Charger le modèle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5  # 5 classes
)

# Tokeniser
def tokenize(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=512)

# Entraîner
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
trainer.train()
```

### Paramètres importants pour BERT
- `max_length=512` : BERT est limité à 512 tokens — les textes plus longs seront tronqués
- `batch_size` : 16 est un bon compromis (réduire si mémoire insuffisante)
- `num_train_epochs` : 3 à 5 epochs suffisent généralement
- `learning_rate` : `2e-5` est la valeur recommandée pour le fine-tuning

---

## Catégorie 5 — Évaluation

**Objectif :** Mesurer les performances du modèle de manière rigoureuse.

### Métriques pour la classification

| Métrique | Ce qu'elle mesure |
|---|---|
| **Accuracy** | % de notes exactement correctes |
| **F1-score (macro)** | Moyenne des F1 par classe (équitable même si déséquilibre) |
| **Matrice de confusion** | Voir quelles notes sont confondues entre elles |

### Métriques pour la régression

| Métrique | Ce qu'elle mesure |
|---|---|
| **MAE** (Mean Absolute Error) | Erreur moyenne en nombre d'étoiles |
| **RMSE** | Pénalise plus les grosses erreurs |

### Ordre de comparaison recommandé

```
TF-IDF + LogReg  →  TF-IDF + SVM  →  BERT extracteur  →  BERT fine-tuné
```

Chaque étape devrait améliorer les résultats. Comparer toujours sur `valid`.

---

## Catégorie 6 — Ordre de travail recommandé

Voici l'ordre conseillé pour avancer progressivement :

### Étape 1 — Exploration (1-2h)
- [ ] Charger les 3 fichiers CSV
- [ ] Afficher la distribution des notes
- [ ] Calculer la longueur moyenne des textes
- [ ] Lire quelques exemples par note

### Étape 2 — Baseline TF-IDF (2-3h)
- [ ] Nettoyer les textes
- [ ] Vectoriser avec TF-IDF
- [ ] Entraîner Logistic Regression
- [ ] Évaluer sur `valid` → noter le score de référence

### Étape 3 — BERT extracteur de features (2-3h)
- [ ] Installer `transformers` et `torch`
- [ ] Charger `bert-base-uncased`
- [ ] Extraire les vecteurs `[CLS]` pour chaque texte
- [ ] Entraîner un classifieur sur ces vecteurs
- [ ] Comparer avec la baseline

### Étape 4 — Fine-tuning BERT (4-6h)
- [ ] Créer un `Dataset` PyTorch avec tokenisation
- [ ] Configurer `BertForSequenceClassification`
- [ ] Entraîner avec `Trainer` de HuggingFace
- [ ] Évaluer sur `valid`
- [ ] Comparer avec les étapes précédentes

### Étape 5 — Prédiction finale (30min)
- [ ] Prendre le meilleur modèle
- [ ] Prédire sur `test_hotel_reviews.csv`
- [ ] Sauvegarder les prédictions

---

## Dépendances à installer

```bash
pip install pandas scikit-learn transformers torch datasets
```

Pour vérifier que PyTorch détecte un GPU (fortement recommandé pour BERT) :
```python
import torch
print(torch.cuda.is_available())  # True si GPU disponible
```

> Sans GPU, le fine-tuning BERT sera très lent. Utiliser Google Colab (gratuit, GPU disponible) est une bonne alternative.

---

## Structure de fichiers recommandée

```
projet/
├── train_hotel_reviews.csv
├── valid_hotel_reviews.csv
├── test_hotel_reviews.csv
├── README.md
├── 1_exploration.ipynb          # Catégorie 1
├── 2_baseline_tfidf.ipynb       # Catégories 2 + 4a
├── 3_bert_features.ipynb        # Catégorie 3 option B1
├── 4_bert_finetuning.ipynb      # Catégorie 3 option B2 + 4b
└── results/                     # Modèles sauvegardés
```
