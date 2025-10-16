## 1) Executive Summary

### What this document delivers

We frame market “forecasts” from options as an **Expected Move (EM)** and compare them to what actually happens, the **Realized Move (RM)**. We then build a **Random Forest Regressor** that predicts the magnitude of RM for a fixed horizon (h) and analyze where forecasts help.

You get simple definitions, leakage controls, feature menus, **time-aware validation**, and a minimal code pipeline. The plan emphasizes honest out-of-sample testing and clear storytelling for class or video use.

### Key takeaways

- Core idea: **forecast the size** of future price movement and compare it to what options already imply.
    
- Modeling approach: **Random Forest Regression** on tabular features (options, price history, calendar).
    
- Evaluation: **walk-forward splits**, **MAE/RMSE** on original scale, plus error-by-bucket and regime checks.
    

---

## 2) Options Primer (Calls, Puts, IV, Time Value)

### Core concepts

An **option** is a right (not obligation) to buy or sell a stock at strike ($K$) by time ($T$). A **call** is a right to buy; a **put** is a right to sell.  
Value splits into **intrinsic value** (how “in the money” now) and **time value** (uncertainty left until expiration).  
**Implied volatility (IV)** is the volatility level that makes a pricing model match today’s option price; higher IV = market expects larger swings.

### Tiny formulas (LaTeX)

- Call payoff: ($\max(S_T - K, 0)$)
    
- Put payoff: ($\max(K - S_T, 0)$)
    

### Why this matters later

Options embed a crowd forecast for **how big** the next move might be. We turn that into an **Expected Move** and ask if we can forecast the **actual** move well enough to add insight.

---

## 3) Expected vs. Realized Move (Definitions + Analogies + Company X Example)

### Definitions (LaTeX where helpful)

- **Expected Move ($EM$):** typical move implied by options for horizon (h). A common rule of thumb using at-the-money IV is $$\text{EM}_{t,h} \approx \sigma^{\text{ATM}}_t \sqrt{\frac{h}{252}} .$$
    A straddle proxy (ATM call + put) divided by spot is another practical estimate.
    
- **Realized Move (RM):** actual move over that horizon. For 1 day:  
    $$  
    \text{RM}_{t,1\text{D}} = \left|\frac{P_{t+1} - P_t}{P_t}\right|.  
    $$  
    For (h) days, many use absolute log-return:$$\text{RM}_{t,h} = \left|\ln\left(\frac{P_{t+h}}{P_t}\right)\right|.$$


### Everyday analogies

- **Weather:** EM is the forecasted rainfall; RM is the water you measure in the gauge.
    
- **Shipping:** EM is the ETA; RM is the actual door-to-door time.
    

### Company X mini-example

Stock at 100 USD; options imply ($\pm5\%$) for next week (($\text{EM}_{t,5\text{D}}=5\%$)):
 
If next week ends at 108 USD, $\text{RM}=8\%$ (larger than expected).
If it ends at 103 USD, $\text{RM}=3\%$ (smaller than expected).

Using high–low over the week increases RM; longer windows smooth it.

---
## 4) Random Forest for This Problem (Concepts, Hyperparameters, When/Why)

### Concept in one minute

A **Random Forest** averages many decision trees trained on bootstrapped samples. Each split considers a random subset of features. Averaging reduces variance and improves stability (**bias–variance** trade-off) without heavy preprocessing.

### What to tune

- `n_estimators` (more trees → smoother predictions)
    
- `max_depth`, `max_features` (control complexity and diversity)
    
- `min_samples_split`, `min_samples_leaf` (regularize small leaves)
    

### Why it fits here

Financial features are **tabular, correlated, and nonlinear**. Forests handle interactions (e.g., “near earnings” × “high short-term IV”) and are robust to scaling and moderate outliers.  
Key risks: **leakage** (using future info by accident) and **non-stationarity** (regimes change). We address both with alignment checks and walk-forward validation.

---
## 5) Data & Labels (Assumptions, Windows, Leakage Controls)

### Assumptions (defaults)

- $EM$ from **ATM IV** or **ATM straddle/spot** for the same horizon ($h$).
    
- $RM$ as **absolute log-return** over ($h$) days.
    
- All features built **only** from information known at time ($t$).
    

### Label design (comparison)

| Task                   | Label Definition                                       | Target Example      |
| ---------------------- | ------------------------------------------------------ | ------------------- |
| Classification         | $\mathbf{1}{\text{RM}_{t,h} \ge \text{EM}_{t,h}}$      | Exceedance flag     |
| Regression (used here) | $\text{RM}_{t,h}$ or $\text{RM}_{t,h}-\text{EM}_{t,h}$ | Magnitude or “edge” |

_For this project we model **regression** (magnitude, or edge = RM–EM)._

### Leakage controls

- No post-(t) quotes/IV or future events in features.
    
- Join by exact timestamps and **lag** where needed.
    
- Re-audit after merges (spot-check random rows).
---

## 6) Feature Engineering (Options-, Underlying-, Calendar-based)

### Options-derived features

- **IV level & term structure:** ( $\sigma^{\text{ATM}}_{7\text{D}}, \sigma^{\text{ATM}}_{30\text{D}}, \Delta\text{(term)}$ ).
    
- **Skew/smile:** 25-delta put IV – call IV; call–put IV gap near ATM.
    
- **Flow & liquidity:** put/call **volume** ratio, ( $\Delta$ )open interest, bid–ask % spread.
    
- **Price proxies:** ATM **straddle/spot** ratio.
    

### Underlying-derived features

- Recent returns $r_{1\text{D}}, r_{5\text{D}}, r_{20\text{D}}$; momentum sign.
    
- **Realized volatility** windows (rolling std), **ATR**, overnight gap %.
    
- Liquidity: volume, turnover (optional microstructure spreads).
    

### Calendar & events

- Earnings **flag** and **days-to-earnings**.
    
- Day-of-week, month dummies; optional macro flags if available.
    

---

## 7) Validation & Metrics (Time-Series CV, Metrics, Robustness)

### Time-aware splits

Use **expanding** or **rolling** **TimeSeriesSplit**. Example: train 2019–2021 → validate 2022; then train 2019–2022 → validate 2023. Avoid random K-fold.

### Metrics to report

- **Regression:** **MAE** (robust and human-readable), **RMSE** (penalizes big misses), ($R^2$) (explanatory power).
    
- Optional volatility scoring: **QLIKE** on ($\text{RM}$).
    

### Robustness checks

- Walk-forward across calm vs. volatile years.
    
- Alternate EM constructions (IV vs straddle/spot).
    
- Error analysis by **EM buckets** (small/medium/large) and by sector.
    

---

## 8) Jupyter Project Pipeline (Step-by-Step with Code Skeletons)

Below are **minimal, runnable** placeholders (no downloads). Replace column names with yours.

1. **Data import & schema checks**
    

```python
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your pre-joined table here
df = pd.DataFrame()  # TODO: replace with your load step

REQ = {'date','ticker','close','iv_atm','straddle_price'}
missing = REQ - set(df.columns)
assert not missing, f"Missing columns: {missing}"

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker','date']).drop_duplicates(['ticker','date'])
```

2. **Cleaning & alignment (timestamps, corporate actions)**
    

```python
# TODO: adjust for splits/dividends if needed
# Ensure options snapshot and stock close correspond to the same decision time t
```

3. **Leakage audit**
    

```python
# Ensure no feature uses post-t info; spot-check by comparing shifted columns
```

4. **Feature engineering functions**
    

```python
def add_price_features(d, h=5):
    d = d.copy()
    d['ret_1d'] = d.groupby('ticker')['close'].pct_change()
    d['rv_5d']  = (d.groupby('ticker')['ret_1d']
                    .rolling(5).std().reset_index(level=0, drop=True))
    d['mom_5d'] = d.groupby('ticker')['close'].pct_change(5)
    return d

def expected_move_from_iv(d, h=5):
    d = d.copy()
    d['em_h'] = d['iv_atm'] * np.sqrt(h/252.0)     # IV-based EM proxy
    return d

def realized_move(d, h=5):
    d = d.copy()
    d['close_fwd'] = d.groupby('ticker')['close'].shift(-h)
    d['rm_h'] = (np.log(d['close_fwd']/d['close'])).abs()
    return d

df = add_price_features(df, h=5)
df = expected_move_from_iv(df, h=5)
df = realized_move(df, h=5)
```

5. **Time-aware splits**
    

```python
# Drop rows with undefined future (last h days)
df = df.dropna(subset=['rm_h','em_h','rv_5d','mom_5d'])

# Choose target: magnitude (RM) or "edge" (RM - EM)
TARGET = 'rm_h'           # or 'edge_h' after the next line
# df['edge_h'] = df['rm_h'] - df['em_h']

FEATURES = ['iv_atm','em_h','rv_5d','mom_5d','straddle_price']  # extend with your features
X = df[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0.0)
y = df[TARGET]
tscv = TimeSeriesSplit(n_splits=5)
```

6. **Baselines (naïve, linear, last-value)**
    

```python
# Naïve: predict y_hat = em_h (for TARGET=rm_h) or 0 (for edge_h)
```

7. **Random Forest training & tuning**
    

```python
rf = RandomForestRegressor(
    n_estimators=500, max_depth=None, max_features='sqrt',
    min_samples_leaf=20, n_jobs=-1, random_state=42
)

mae_list, rmse_list = [], []
for tr_idx, te_idx in tscv.split(X):
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    rf.fit(Xtr, ytr)
    p = rf.predict(Xte)
    mae_list.append(mean_absolute_error(yte, p))
    rmse_list.append(mean_squared_error(yte, p, squared=False))

print("MAE (mean over folds):", np.mean(mae_list))
print("RMSE (mean over folds):", np.mean(rmse_list))
```

8. **Evaluation & calibration (regression)**
    

```python
# Plotting omitted here; recommended: predicted vs actual scatter and residual hist
# Also report errors by EM terciles:
df_eval = df.iloc[te_idx].copy()
df_eval['pred'] = p
df_eval['em_bucket'] = pd.qcut(df_eval['em_h'], 3, labels=['low','mid','high'])
print(df_eval.groupby('em_bucket')
      .apply(lambda g: pd.Series({
          'MAE': mean_absolute_error(g[TARGET], g['pred']),
          'RMSE': mean_squared_error(g[TARGET], g['pred'], squared=False)
      })))
```

9. **Error analysis (ticker, regime, EM buckets)**
    

```python
# Group by year or volatility regime; inspect where errors spike
df_eval['year'] = df_eval['date'].dt.year
print(df_eval.groupby('year')[['pred', TARGET]].apply(
    lambda g: pd.Series({
        'MAE': mean_absolute_error(g[TARGET], g['pred'])
    })
))
```

10. **Feature importance & optional SHAP-style views**
    

```python
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances)
```

11. **Packaging results (tables/plots)**
    

```python
# Save metrics/figures to a results/ folder; include seed and split dates in filenames
```

12. **Reproducibility (seeds, config, artifacts)**
    

```python
CONFIG = {'h':5, 'seed':42, 'features':FEATURES}
# Save CONFIG and model params to disk alongside results
```

---

## 12) Pros, Cons, and Suitability as a Learning Showcase

### Strengths

Works well on **tabular** data, captures **nonlinear** patterns, needs little scaling, and yields intuitive feature rankings. A clean fit for a data-mining class and a clear YouTube demo.

### Limitations

Markets are **non-stationary**; edges shift across regimes. Labels depend on how **EM** is built. Noise from illiquid options (wide spreads) can cap accuracy.

### When it shines vs. when it doesn’t

Shines as a rigorous **pipeline example** with honest out-of-sample tests and clear visual explanations. It is _not_ a live trading system without cost models and risk controls.

---

## 13) Risks, Caveats, and Next Steps

### Educational disclaimer

This is for learning, not financial advice.

### Reproducibility

Fix random seeds; store a `config.yaml`; save metrics/plots with timestamps.

### Next experiments

Try horizons ( $h \in {1,5,10}$ ). Compare to **Ridge** and **Gradient Boosting**. Add **quantile prediction** (prediction intervals) and monitor coverage.

---

## 14) Glossary (Brief, Precise)

- **Call:** Right to buy at (K) by (T); payoff ( $\max(S_T-K,0)$ ).
    
- **Put:** Right to sell at (K) by (T); payoff ( $\max(K-S_T,0)$ ).
    
- **Implied Volatility (IV):** Vol level that makes a pricing model match today’s option price.
    
- **Expected Move (EM):** Move implied by options for horizon ($h$), e.g.,$\sigma^{\text{ATM}}\sqrt{h/252}$.
    
- **Realized Move (RM):** Actual move over (h), e.g., ( $\left|\ln(P_{t+h}/P_t)\right|$ ).
    
- **Calibration (regression):** Agreement between predicted and actual magnitudes.
    
- **Walk-forward validation:** Train on past, test on later periods in rolling windows.
    
- **Permutation importance:** Drop in score when a feature’s values are shuffled—measures dependence.
