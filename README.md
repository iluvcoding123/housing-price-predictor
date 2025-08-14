# Housing Price Predictor

An end-to-end machine learning pipeline for predicting home sale prices using the Ames Housing dataset.  
Includes:
- Data cleaning, preprocessing, and feature engineering
- Model training, tuning, and evaluation (Linear Regression, Random Forest, XGBoost)
- Deployment as a Flask REST API (Dockerized)
- Optional cloud deployment (AWS-ready)

## Project Structure
```
housing-price-predictor/
├── data/               # Raw and processed data
├── docs/               # Documentation
├── models/             # Saved models and scaler
├── notebooks/          # EDA, preprocessing, modeling notebooks
├── src/                # API and prediction scripts
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset
- **Source**: Ames Housing Dataset (Kaggle)
- **Size**: 2,930 observations, 80+ features
- **Target**: `SalePrice` (USD)
- Includes numerical, categorical, and date features.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Flask (REST API)
- Docker
- Matplotlib & Seaborn (Visualization)
- Joblib (Model persistence)

## Workflow
1. **EDA** – Explore dataset, visualize relationships, and detect outliers.
2. **Feature Engineering** – Create new features like `TotalSF` (Total Square Footage).
3. **Preprocessing** – Handle missing values, encode categoricals, scale numerics.
4. **Modeling** – Train and evaluate multiple regression models.
5. **Deployment** – Save best model + scaler, serve predictions via Flask API.

## Model Performance
| Model              | RMSE     | MAE     | R²     |
|--------------------|----------|---------|--------|
| Linear Regression  | 33,926.91| 24,304.41 | 0.8013 |
| Random Forest      | 30,574.59| 21,173.68 | 0.8471 |
| XGBoost (Best)     | 30,389.81| 21,186.84 | 0.8492 |

## Local Setup (Non-Docker)
git clone https://github.com/iluvcoding123/housing-price-predictor.git
cd housing-price-predictor
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/api.py

## Run the API in Docker
```bash
docker build -t housing-price-predictor .
docker run -p 8000:8000 --name hpp-api housing-price-predictor
# Health
curl http://127.0.0.1:8000/health
# Predict
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[{"Gr Liv Area":1800,"Total Bsmt SF":900,"Garage Cars":2,"Overall Qual":7,"Year Built":2005,"Yr Sold":2010}]}'