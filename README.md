# 🏋️ Calories Burn Prediction (Kaggle Playground S5E5)

This project was developed for the **[Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview)** competition.  
The goal is to predict the calories burned based on physiological and activity data, achieving the lowest possible error.

---

## 📂 Project Steps

### 1️⃣ Exploratory Data Analysis (EDA)
- Train: `(750,000 x 9)`  
- Test: `(250,000 x 8)`  
- No missing values, categorical column: `Sex`  
- Numerical features: `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- Target variable: `Calories`
- Correlation analysis → highest correlation with **`Duration`** and **`Heart_Rate`**

---

### 2️⃣ Feature Engineering (FE)
- **BMI** = `Weight / (Height/100)^2`
- **Intensity** = `Duration * Heart_Rate`
- `log1p(Calories)` transformation → compatible with RMSLE / MSLE metric
- Numerical & categorical preprocessing via **ColumnTransformer**:
  - Numerical: `passthrough`  
  - Categorical: One-Hot Encoding (`drop='first'`)

---

### 3️⃣ Modeling
- Main model: **XGBoost Regressor**
- Evaluation metric: **RMSLE**
- **StratifiedKFold** (5-fold) → target values binned after log1p for stratification
- **Early Stopping**: stop if no improvement for 100 rounds

---

### 4️⃣ Hyperparameter Optimization
- **Optuna** for parameter search (20 trials)
- Parameters tuned:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `min_child_weight`
  - `subsample`
  - `colsample_bytree`
  - `reg_lambda`, `reg_alpha`
  - `gamma`
- Best parameter set significantly reduced CV RMSLE.

---

### 5️⃣ Results
| Step | CV RMSLE | Kaggle Public LB | Kaggle Private LB |
|------|----------|------------------|-------------------|
| Ridge (baseline) | 0.1799 | - | - |
| XGBoost (OHE) | 0.0603 | - | - |
| XGBoost + Optuna + FE | **0.0116** | **0.05935** | **0.05805** |

---

### 6️⃣ Key Takeaways
- **log1p / expm1 transformation** is critical when working with RMSLE
- **Feature engineering** (BMI, Intensity) led to significant error reduction
- **StratifiedKFold** preserved the distribution of small target values for stable CV
- CV and Kaggle LB scores may differ → hold-out validation is essential
- More Optuna trials may yield better results but increase runtime

---

