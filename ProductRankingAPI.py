from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.post("/rank-products")
def product_ranking(payload: dict):
    produk_list = payload.get("produk")
    bobot = payload.get("bobot")

    # Pisahkan nama dan fitur jadi 2 struktur
    list_nama = [p["nama"] for p in produk_list]
    matriks_fitur = [p["fitur"] for p in produk_list]

    # Hitung skor SEMUA produk sekaligus (vectorized, no loop)
    skor_data = np.dot(matriks_fitur, bobot)

    # Gabungkan lagi jadi ranking, convert eksplisit ke tipe Python native
    rankings = [
        {"nama": nama, "skor": round(float(skor), 2)}
        for nama, skor in zip(list_nama, skor_data)
    ]
    rankings.sort(key=lambda x: x["skor"], reverse=True)

    return {
        "rankings": rankings,
        "statistik": {
            "mean": round(float(np.mean(skor_data)), 2),
            "std": round(float(np.std(skor_data)), 2)
        }
    }