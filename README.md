# 🧠 Prédiction du Départ des Employés avec l'IA Explicable

## 📌 Présentation du projet

Ce projet consiste à développer une application web interactive permettant de **prédire le risque de départ des employés** d'une entreprise à l'aide d'algorithmes de Machine Learning.

L'objectif est également d'expliquer les prédictions des modèles grâce aux techniques d'**intelligence artificielle explicable (XAI)**, notamment SHAP et LIME.

L'application repose sur une architecture modulaire avec **FastAPI pour le backend** et **Streamlit pour le frontend**.

## 🎯 Problématique métier

**Comment une entreprise peut-elle anticiper le départ de ses employés et identifier les facteurs influençant leur décision de quitter l'organisation ?**

L'application vise à aider les responsables RH à mieux comprendre les facteurs associés à l'attrition et à orienter leurs actions de fidélisation.

## 📊 Jeu de données

**Dataset :** IBM HR Analytics Employee Attrition & Performance

**Source :** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- 1 470 employés
- 35 variables, cible comprise
- Classification binaire : `Attrition` (Yes / No)
- Données RH synthétiques

Quelques variables importantes :
- Âge et ancienneté
- Salaire mensuel
- Satisfaction professionnelle
- Heures supplémentaires
- Équilibre vie professionnelle / personnelle
- Poste occupé

Les données seront récupérées automatiquement via **KaggleHub**, sans téléchargement manuel.

## 🛠️ Technologies utilisées

| Domaine | Technologies |
|---|---|
| Langage | Python |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Traitement des données | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| IA explicable | SHAP, LIME |
| Visualisation | Plotly, Matplotlib |
| Source de données | KaggleHub |
| Versioning | Git, GitHub |

## 🏗️ Architecture du projet

```text
employee-attrition-xai/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/          # Endpoints REST
│   │   ├── services/     # Traitements ML et XAI
│   │   └── schemas/      # Validation des données
│   └── artifacts/        # Modèles entraînés
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── pages/            # Pages Streamlit
│
├── data/                 # Données locales
├── notebooks/            # Expérimentations
├── tests/                # Tests
├── reports/              # Rapport final
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Fonctionnement

1. Le frontend Streamlit permet à l'utilisateur d'interagir avec l'application.
2. Les requêtes sont envoyées aux endpoints FastAPI.
3. Les endpoints appellent les services Python du backend.
4. Les services réalisent les traitements : preprocessing, entraînement, évaluation et explicabilité.
5. Les résultats sont renvoyés au frontend pour visualisation.

## ⚙️ Fonctionnalités prévues

### 1. Exploration des données
- Importation automatique depuis Kaggle ou d'un fichier CSV.
- Affichage des statistiques descriptives.
- Analyse des valeurs manquantes.
- Visualisations interactives.

### 2. Préparation des données
- Sélection des variables.
- Gestion des valeurs manquantes.
- Encodage des variables catégorielles.
- Normalisation.
- Séparation entraînement/test.

### 3. Entraînement des modèles
- Régression logistique
- Arbre de décision
- KNN
- Random Forest
- XGBoost

Avec possibilité de sélectionner les modèles et de configurer leurs hyperparamètres.

### 4. Évaluation des performances
- Précision, rappel et F1-score.
- ROC-AUC et PR-AUC.
- Matrices de confusion.
- Courbes ROC et précision-rappel.
- Comparaison des modèles.
- Ajustement interactif du seuil de décision.

### 5. Intelligence artificielle explicable
- Importance des variables.
- Importance par permutation.
- Explications globales avec SHAP.
- Explications individuelles avec SHAP et LIME.
- Interprétation des prédictions dans un langage compréhensible.

### 6. API REST
Une API FastAPI permettra d'exposer les services de traitement et de prédiction au frontend.

## 🚀 Installation et exécution

### 1. Cloner le projet

```bash
git clone https://github.com/sarrabouraoui21/employee-attrition-xai.git
cd employee-attrition-xai
```

### 2. Créer un environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

### 4. Lancer le backend

Depuis la racine du projet :

```powershell
python -m uvicorn app.main:app --app-dir backend --reload --log-level debug --access-log
```

- API : http://localhost:8000
- Documentation Swagger : http://localhost:8000/docs

### 5. Lancer le frontend

Dans un deuxième terminal :

```powershell
python -m streamlit run frontend/app.py
```

- Application : http://localhost:8501

**Remarque :** les commandes de lancement seront opérationnelles une fois les composants développés.

## 🔍 Logs et gestion des erreurs

Le backend devra afficher ses logs directement dans le terminal VS Code.

Les logs permettront de suivre :
- Les requêtes HTTP.
- Le chargement des données.
- Les étapes du preprocessing.
- L'entraînement des modèles.
- Les calculs SHAP et LIME.
- Les erreurs et leurs traces détaillées.

Les données personnelles ne devront pas être exposées dans les logs.

## 📦 Livrables

- Application Streamlit interactive.
- API FastAPI documentée.
- Modèles de classification et comparaison.
- Explications SHAP et LIME.
- Code source sur GitHub.
- Notebook d'expérimentation.
- Fichier `requirements.txt`.
- Application et API déployées.
- Rapport de synthèse d'une page (PDF).

## ⚠️ Limites

Le dataset utilisé contient des données synthétiques. Les résultats ne peuvent donc pas être généralisés directement à de véritables employés.

Les explications SHAP et LIME mettent en évidence les facteurs influençant les prédictions, mais ne démontrent pas de relations causales.

Les risques de biais et les questions d'équité doivent être pris en compte dans les applications RH.

## 🎓 Contexte académique

- **Établissement :** ENSI
- **Module :** Data Mining
- **Projet :** Challenge 1 — Application Web Interactive de Classification Explicable
- **Année universitaire :** 2026–2027
- **Enseignante :** Rym Besrour

## 📌 État du projet

🚧 En cours de développement.
