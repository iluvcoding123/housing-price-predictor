# housing-price-predictor
End-to-end machine learning pipeline to predict housing prices. Includes data preprocessing, model training, deployment, and cloud integration.

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