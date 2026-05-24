# Predicting P/E Ratios of Nifty 500 Companies

**A quantitative equity-valuation model that predicts Price-to-Earnings ratios from financial fundamentals, ownership structure, and market characteristics — and identifies where the market systematically deviates from those fundamentals.**

> **Headline result:** A tuned Gradient Boosting model explains **~84% of the variation** in (log) P/E ratios across 330 Nifty 500 companies (out-of-fold R² = 0.84), with a low generalization gap of 0.12 — meaning it holds up on unseen data rather than memorizing the training set.

---

## Why this project

A company's P/E ratio is one of the most-watched valuation signals in equity markets, yet it is notoriously hard to explain from fundamentals alone — two firms with similar earnings can trade at wildly different multiples. This project asks two questions a quant analyst actually cares about:

1. **How much of a company's P/E can be explained by its fundamentals?** (profitability, growth, ownership, balance-sheet structure, market risk)
2. **Where does the market price companies above or below what fundamentals justify** — and what does that gap tell us about each sector?

The second question is the interesting one. The model's *errors* are not just noise; grouped by sector, they reveal a systematic valuation structure.

---

## Key finding: sectors carry persistent valuation premiums

After fitting the model, I analyzed the residuals (Actual P/E − Predicted P/E) by sector. A positive residual means the market assigns a **premium** beyond what balance-sheet variables explain; a negative residual means a **discount**.

| Sector | Mean residual (P/E points) | Interpretation |
|---|---|---|
| Information Technology | **+3.3** | Largest premium — intangibles, brand and pricing power not captured by balance-sheet inputs |
| Consumer Staples | +2.8 | Defensive, predictable earnings command a premium |
| Consumer Discretionary | +2.8 | Growth optionality priced in |
| Healthcare | +2.5 | R&D pipeline / regulatory moats |
| Materials | **+1.2** | Smallest premium — cyclicality and commodity exposure |

The IT sector trades at the largest premium to model predictions, consistent with brand and pricing-power intangibles that don't appear on the balance sheet. Cyclical sectors like Materials show the thinnest premiums. **This is the kind of insight that turns a prediction model into a research tool.**

---

## Approach

The notebook follows a deliberate, leakage-aware workflow:

**1. Raw exploratory analysis** — distributions, missing values, and outlier diagnosis *before* any cleaning, so every cleaning decision is justified by evidence.

**2. Target treatment** — raw P/E was severely right-skewed (skewness ≈ 18). I removed invalid P/E = 0 rows (25 firms), trimmed IQR outliers, and applied a `log(1 + P/E)` transform, bringing skewness to ≈ -0.5 (near-normal). Final sample: **330 companies**.

**3. Feature engineering** — six economically-motivated features beyond the raw inputs:

| Feature | Formula | Rationale |
|---|---|---|
| Profit Margin | Profit / Sales | Operating efficiency |
| R&D Intensity | R&D / Sales | Innovation investment (predictive in tech & pharma) |
| Other-Income Ratio | Other Income / Profit | Earnings quality |
| Growth Momentum | mean(Sales, Profit, BV growth) | Smoothed growth signal |
| Promoter Confidence | Promoter Holding − Pledged | Governance / management conviction |
| Book-Value Yield | BVPS / P/E | Value indicator |

**4. Multicollinearity handling** — dropped `Total_Expenses` (r = 1.00 with Sales), `Total_Equity` (r = 0.80 with Profit), and `Institutional_Holding` (r = -0.81 with Promoter Holding) to keep coefficients interpretable and reduce variance.

**5. Robust evaluation** — 5-fold cross-validation **stratified jointly by P/E bin and sector**, so every fold is representative on both dimensions. All metrics reported are out-of-fold (no data leakage).

**6. Model competition** — benchmarked 18 regressors (linear, SVM, tree ensembles, gradient boosting, neural nets), then tuned the top performers with `RandomizedSearchCV` and tested voting and stacking ensembles.

---

## Results

| Metric | Value |
|---|---|
| Best model | Gradient Boosting (tuned) |
| Out-of-fold R² | **0.84** |
| RMSE (log space) | 0.26 |
| MAE (log space) | 0.17 |
| Generalization gap (train R² − val R²) | 0.12 |

The low generalization gap is as important as the R² itself: it shows the model generalizes rather than overfits — the result of deliberate regularization (shallow trees, subsampling, L1/L2 penalties) across every ensemble.

---

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Matplotlib` · `Seaborn` · `SciPy`

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── nifty500_features.csv            # Modeling dataset (fundamentals + sectors)
│   └── nifty500_company_indicators.csv  # Company reference (CIN, ISIN, NSE symbol, industry)
└── notebooks/
    └── pe_ratio_prediction.ipynb        # Full EDA → modeling → residual analysis pipeline
```

---

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/nifty500-pe-prediction.git
cd nifty500-pe-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/pe_ratio_prediction.ipynb
```

The notebook runs top-to-bottom and uses relative paths, so it works out of the box after cloning.

---

## Data

**Source:** [CMIE Prowess](https://prowess.cmie.com/) — a financial database of Indian companies maintained by the Centre for Monitoring Indian Economy.

The dataset covers **Nifty 500 constituents** with FY2024 financials: ownership (promoter holding, pledging, institutional holding), income-statement items (sales, profit, expenses, other income, R&D), balance-sheet metrics (book value per share, total equity), growth rates, and market structure (Beta, stock return, P/E), labeled by GICS sector. Company identifiers (CIN, ISIN, NSE symbol, NIC industry codes) are included for reference and cross-linking.

---

## What I'd build next

- Add a time dimension (multi-year panel) to separate *cyclical* from *structural* sector premiums.
- Test SHAP values for per-company valuation attribution — useful as a screening tool for over/undervalued names.
- Wrap the model in a small dashboard that flags companies trading furthest from their fundamental fair value.

---

*Built as a quantitative equity-research project bridging machine learning and fundamental valuation.*
