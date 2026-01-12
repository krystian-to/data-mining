# Telecom Churn – Data Mining Project

Projekt z Data Mining polegający na porównaniu 3 modeli klasyfikacyjnych
dla problemu churnu klientów.

## Modele

- Decision Tree
- XGBoost
- Neural Network

## Struktura projektu

- `data/raw` – dane źródłowe
- `data/processed` – przetworzone dane (train / val / test)
- `notebooks` – notebooki do trenowania modeli
- `src` – wspólny kod (config, preprocessing, metryki)
- `results` – zapis wyników i metryk

## Workflow

1. `00_data_prep.ipynb` – preprocessing i zapis danych do `data/processed`
2. Notebooki modeli:
   - `10_decision_tree.ipynb`
   - `20_xgboost.ipynb`
   - `30_neural_net.ipynb`
3. Każdy notebook zapisuje:
   - metryki do `results/`
   - wykresy do `results/`

## Uruchomienie

```bash
pip install -r requirements.txt
```
