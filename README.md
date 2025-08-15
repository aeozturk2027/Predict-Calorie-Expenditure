# 🏋️ Calories Burn Prediction (Kaggle Playground S5E5)

Bu proje, Kaggle'ın **[Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview)** yarışması için geliştirilmiş bir **kalori tahmin modeli** içerir.  
Amaç, verilen fizyolojik ve aktivite verilerinden yola çıkarak yakılan kaloriyi mümkün olan en düşük hata ile tahmin etmektir.

---

## 📂 Proje Adımları

### 1️⃣ Veri Keşfi (EDA)
- Train: `(750,000 x 9)`  
- Test: `(250,000 x 8)`  
- Eksik veri yok, kategorik sütun: `Sex`  
- Sayısal özellikler: `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- Hedef değişken: `Calories`
- Korelasyon analizi → en yüksek korelasyon **`Duration`** ve **`Heart_Rate`** ile

---

### 2️⃣ Feature Engineering (FE)
- **BMI** = `Weight / (Height/100)^2`
- **Intensity** = `Duration * Heart_Rate`
- `log1p(Calories)` dönüşümü → RMSLE / MSLE ile uyumlu hata ölçümü
- Sayısal ve kategorik veriler **ColumnTransformer** ile işlendi  
  - Sayısal: `passthrough`  
  - Kategorik: One-Hot Encoding (`drop='first'`)

---

### 3️⃣ Modelleme
- Ana model: **XGBoost Regressor**
- Hedef hata metriği: **RMSLE**
- **StratifiedKFold** (5-fold) kullanıldı → hedef değer log1p sonrası bin’lenerek stratify edildi
- **Early Stopping**: 100 tur gelişme olmazsa durdurma

---

### 4️⃣ Hiperparametre Optimizasyonu
- **Optuna** ile parametre arama (20 trial)
- Aranan parametreler:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `min_child_weight`
  - `subsample`
  - `colsample_bytree`
  - `reg_lambda`, `reg_alpha`
  - `gamma`
- En iyi parametre seti CV RMSLE’de ciddi düşüş sağladı.

---

### 5️⃣ Sonuçlar
| Adım | CV RMSLE | Kaggle Public LB | Kaggle Private LB |
|------|----------|------------------|-------------------|
| Ridge (baseline) | 0.1799 | - | - |
| XGBoost (OHE) | 0.0603 | - | - |
| XGBoost + Optuna + FE | **0.0116** | **0.05935** | **0.05805** |

---

### 6️⃣ Çıkarımlar
- **log1p / expm1 dönüşümü** RMSLE uyumlu tahmin için kritik
- **Feature engineering** (BMI, Intensity) hatayı ciddi düşürdü
- **StratifiedKFold** küçük target değerlerini koruyarak daha kararlı CV sağladı
- CV ve Kaggle skorları farklı çıkabilir → mutlaka hold-out testi yapılmalı
- Hiperparametre optimizasyonunda fazla trial daha iyi sonuç verebilir ama zaman maliyeti var

---

## 🚀 Kullanım

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Eğitim ve tahmin
python train.py

# Submission dosyası oluşturma
python predict.py
