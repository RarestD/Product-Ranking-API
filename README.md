# Product Ranking API

A lightweight REST API built with **FastAPI** and **NumPy** that ranks products based on weighted feature scores — useful as the scoring engine behind an e-commerce recommendation or sorting system.

## Features

- Vectorized scoring using NumPy matrix multiplication — no manual `for` loops for the actual computation
- Configurable weights per feature (e.g. rating, popularity, freshness)
- Returns a ranked product list plus summary statistics (mean & standard deviation of scores)

## Tech Stack

- Python 3
- FastAPI
- NumPy
- Uvicorn (ASGI server)

## How It Works

Each product is represented as a feature vector, e.g. `[rating, popularity, freshness]`. The API computes a weighted score for every product in one shot using the dot product between the feature matrix and a weight vector, then returns the products sorted from highest to lowest score — the same core operation used inside a single layer of a neural network.

## Installation & Running Locally

```bash
pip install fastapi uvicorn numpy
uvicorn ProductRankingAPI:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

## API Reference

### `POST /rank-products`

**Request body**

```json
{
  "produk": [
    {"nama": "Kaos A", "fitur": [4.5, 0.8, 0.9]},
    {"nama": "Kaos B", "fitur": [3.2, 0.6, 0.5]}
  ],
  "bobot": [0.6, 0.3, 0.1]
}
```

**Response**

```json
{
  "rankings": [
    {"nama": "Kaos A", "skor": 3.03},
    {"nama": "Kaos B", "skor": 2.19}
  ],
  "statistik": {"mean": 2.61, "std": 0.42}
}
```

## Lessons Learned

- Replaced a manual per-product `for` loop with a single vectorized `matrix @ weights` operation to score all products simultaneously, instead of looping one product at a time.
- Learned to explicitly cast NumPy scalar types (`np.float64`) to native Python types (`float(...)`) before returning them in a JSON response, to avoid subtle serialization issues that only surface with certain numeric dtypes.

## Possible Improvements

- Request validation with Pydantic models instead of accepting a raw `dict`
- Explicit tie-breaking rule for products with identical scores
- Support for saving/reusing weight configurations per client use case
