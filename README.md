# 🛒 Predictive Sales Analytics — XGBoost + SHAP

> Explainable retail sales forecasting using gradient boosting and SHapley Additive exPlanations.  
> Based on the research paper: *"Predictive Sales Analytics Using Machine Learning: An XGBoost and SHAP-Based Approach to Explainable Retail Forecasting"*

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **98.17%** |
| R² | **0.9999** |
| MAE | **Rs 2.91** |
| RMSE | **Rs 4.23** |

---

## 🧠 Overview

Traditional forecasting methods like ARIMA assume linear, stationary patterns — a poor fit for real-world retail data driven by promotions, demographics, and seasonality. This project trains a regularized **XGBoost regressor** on transaction-level retail data and pairs it with **SHAP** to produce predictions that are both highly accurate and fully interpretable by non-technical stakeholders.

---

## 🗂️ Project Structure
```
PREDICTIVE_SALES_ANALYTICS/
├── myenv/                        # Python virtual environment (excluded from Git)
├── app.ipynb                     # Main Jupyter Notebook — full pipeline
└── retail_sales_dataset.csv      # Raw retail transaction data (from Kaggle)
```

> **Note:** Add `myenv/` and `retail_sales_dataset.csv` to your `.gitignore` before pushing.

---

## 📓 Code Walkthrough — `app.ipynb`

The notebook is organized into the following stages:

### 1. 📥 Data Loading
Reads `retail_sales_dataset.csv` using Pandas and parses the `Date` column into a proper datetime format.
```python
df = pd.read_csv("retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
```

### 2. 🔧 Feature Engineering
Builds 15 input features from the raw transaction data:
```python
# Temporal features
df['Month']      = df['Date'].dt.month
df['DayOfWeek']  = df['Date'].dt.dayofweek
df['Quarter']    = df['Date'].dt.quarter
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
df['Season']     = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,
                                     6:2,7:2,8:2,9:3,10:3,11:3})

# Demographic features
df['AgeGroup'] = np.clip((df['Age'] - 18) // 10, 0, 4)

# Encoded categoricals
df['Gender_enc']   = LabelEncoder().fit_transform(df['Gender'])
df['Category_enc'] = LabelEncoder().fit_transform(df['Product Category'])

# Interaction features
df['PriceQuantityInteraction'] = np.log1p(df['Price per Unit'] * df['Quantity'])
df['AgePriceInteraction']      = df['Age'] * df['Price per Unit']
```

| Group | Features |
|-------|----------|
| **Raw transactional** | `Age`, `Gender_enc`, `Category_enc`, `Quantity`, `Price per Unit` |
| **Temporal** | `Month`, `DayOfWeek`, `Quarter`, `WeekOfYear`, `IsWeekend`, `Season` |
| **Aggregate** | `CatAvgSpend` *(training set only — no data leakage)* |
| **Interaction** | `PriceQuantityInteraction`, `AgePriceInteraction` |

### 3. ✂️ Train / Test Split
Data is split **80/20** with `random_state=42` for full reproducibility.  
`CatAvgSpend` is computed **after** the split from training data only — preventing any leakage into the test set.
```python
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Leak-free category average spend
cat_avg_map = train_df.groupby('Product Category')['Total Amount'].mean().to_dict()
fallback    = train_df['Total Amount'].mean()

train_df['CatAvgSpend'] = train_df['Product Category'].map(cat_avg_map).fillna(fallback)
test_df['CatAvgSpend']  = test_df['Product Category'].map(cat_avg_map).fillna(fallback)
```

### 4. 🤖 Model Training
An `XGBRegressor` is trained with heavy regularization and shallow trees to prevent overfitting across the 15-feature input space.
```python
model = xgb.XGBRegressor(
    n_estimators     = 160,
    max_depth        = 3,      # Shallow trees — reduces overfitting
    learning_rate    = 0.06,
    subsample        = 0.65,   # Row sampling — adds randomness
    colsample_bytree = 0.65,   # Column sampling — adds randomness
    min_child_weight = 12,
    gamma            = 2.0,    # Min loss reduction to split
    reg_alpha        = 2.0,    # L1 regularization
    reg_lambda       = 5.0,    # L2 regularization
    random_state     = 42,
    n_jobs           = -1
)
model.fit(X_train, y_train)
```

> High regularization (`gamma`, `reg_alpha`, `reg_lambda`) and `max_depth=3` are intentional design choices — they keep the model generalizable across the expanded feature space.

### 5. 📈 Evaluation
Evaluated on the held-out 20% test set:
```python
mae      = mean_absolute_error(y_test, y_pred)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
r2       = r2_score(y_test, y_pred)
mape     = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
accuracy = max(0, 100 - mape)
```

| Metric | Value |
|--------|-------|
| R² | 0.9999 |
| MAE | Rs 2.91 |
| RMSE | Rs 4.23 |
| Accuracy | 98.17% |

### 6. 🔍 SHAP Explainability
SHAP is run on a 200-sample subset of the test set to explain every prediction:
```python
explainer   = shap.TreeExplainer(model)
sample      = X_test.sample(min(200, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)
```

Four visualizations are generated:

| Plot | What It Shows |
|------|---------------|
| `summary_plot` | Per-feature SHAP distribution across all test points — magnitude + direction |
| `bar chart` | Ranked mean absolute SHAP values — easy to present to stakeholders |
| `waterfall` | Step-by-step prediction breakdown for one individual transaction |
| `dependence_plot` | How `Price per Unit` interacts with `Quantity` non-linearly |

**Top drivers of predicted sales value (by SHAP):**
```
1. PriceQuantityInteraction   ← strongest signal
2. Price per Unit
3. Quantity
4. Product Category
5. CatAvgSpend
6. Season / Month             ← temporal patterns
7. AgePriceInteraction        ← demographic signal
```

---

## 🧪 Baseline Comparison

| Model | Notes |
|-------|-------|
| Linear Regression | Assumes linearity — fails on non-linear retail patterns |
| Random Forest | Good but lower accuracy than XGBoost |
| ARIMA(1,1,1) | Univariate only — ignores demographics and categories |
| LSTM (64→32 units) | High compute cost, low interpretability |
| **XGBoost (ours)** | **Best across all metrics — accurate and explainable** |

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-username/predictive-sales-analytics.git
cd predictive-sales-analytics
```

### 2. Create and activate virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install numpy pandas xgboost shap scikit-learn jupyter
```

### 4. Add the dataset

Download the retail transactions dataset from [Kaggle](https://www.kaggle.com/) and place it in the project root as:
```
retail_sales_dataset.csv
```

### 5. Launch the notebook
```bash
jupyter notebook app.ipynb
```

Run all cells from top to bottom — training, evaluation, and all four SHAP plots will generate automatically.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `xgboost` | Gradient boosted tree model |
| `shap` | Explainability layer |
| `scikit-learn` | Preprocessing, splitting, metrics |
| `jupyter` | Notebook environment |

---

## 📬 Contact

**Ashim Maity** — maityashim81@gmail.com  
