# mlops_house_price
House price prediction project

## Initial configuration
**For usage:**  
```
pip install miniconda
python -m venv house_prices
cd house_prices
```
- clone the repository
```
cd house-prices
poetry install
pre-commit install
pre-commit run -a
```

## Train model
```
python train.py
```

## Predict on test
```
python infer.py
```
