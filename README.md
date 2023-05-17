# Shale Gas Wells Production Prediction

<a href="https://github.com/georgemuriithi/shale-gas-wells/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/georgemuriithi/shale-gas-wells.svg?color=blue&cachedrop">
</a>

https://www.datascience-contest.com

The <a href="https://www.knoc.co.kr/ENG/main.jsp">Korea National Oil Corporation</a> was interested in purchasing shale gas wells from the United States and wanted to predict their production to select wells that maximize profit.

A combination of **LightGBM regression** and **Exponential smoothing** is used to predict productions. 0-1 integer programming using **Gurobi** is used for optimization to maximize profit. Performance evaluation is based on **sMAPE (symmetric Mean Absolute Percentage Error).** Our team has one of the best performances, having a percentage error of **25.54%,** compared to the best one of **19.49%.**

## Problem Description
### Data

*Unfortunately, the train and exam datasets are confidential. Therefore, they are not included in this repository.*

- **trainSet.csv** - Data of 280 shale gas wells for training models
- **examSet.csv** - Data of 44 shale gas wells for prediction

### Predicting Gas Production
The task is to predict the monthly average gas productions of 44 shale gas wells in **examSet.csv** for the next 6 months.

Performance evaluation is based on **sMAPE (symmetric Mean Absolute Percentage Error):**

<p align="center">
  <img src="https://user-images.githubusercontent.com/21691211/148675936-b3f0def1-44fa-4d76-a9b4-05bc79049fca.png">
</p>

- F<sub>i</sub> - predicted monthly average gas production of i<sup>th</sup> gas well over the next 6 months
- A<sub>i</sub> - actual monthly average gas production of i<sup>th</sup> gas well over the next 6 months
- n - number of gas wells (44 in this problem)

### Investment Decision

A budget of $15,000,000 is given. The task is to select gas wells among the 44 candidates to maximize profit after predicting their monthly average gas productions:

<p align="center">
  <img src="https://user-images.githubusercontent.com/21691211/148675948-b08621d8-68cf-4fa3-82a5-467c3b973347.png">
</p>

- A<sub>i</sub> - actual monthly average gas production of i<sup>th</sup> gas well over the next 6 months
- P<sub>i</sub> - price of i<sup>th</sup> gas well
- P<sub>s</sub> - shale gas price ($5 per 1 Mcf)
- C<sub>i</sub> - monthly operation cost of i<sup>th</sup> gas well
- X<sub>i</sub> - decision variable to purchase i<sup>th</sup> gas well (if purchasing i<sup>th</sup> gas well: X<sub>i</sub> = 1, else: X<sub>i</sub> = 0)

## Solution Approach
The wells are divided into **new wells** and **old wells.** New wells do not have data on gas production, non-gas production and hours operated per month. This data is available for old wells.

Therefore, **regression** is used to predict the monthly average productions of **new wells for the first 6 months**, and **exponential smoothing** is used to predict the monthly average productions of **old wells for the last 6 months.**

### <a href="https://github.com/georgemuriithi/shale-gas-wells/blob/main/New-Wells-Prediction.ipynb">New Wells</a>
<a href="https://colab.research.google.com/drive/1wg5sLr3LeWGhc4oeIqkiocIT4FMsgHwF?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

After **Feature engineering** and **EDA (Exploratory Data Analysis),** the following **advanced decision tree-based models** for **regression** are tested:

- `BaggingRegressor`
  - `n_estimators=50`
- `RandomForestRegressor`
  - `n_estimators=50`
- `XGBRegressor`
  - `max_depth=5`
  - `objective='reg:squarederror'`
- `LGBMRegressor`
- `VotingRegressor`
  - `estimators=[bagging, random_forest, xgb, lgbm]`
  - `n_jobs=-1`

**Hyperparameter:** `train_test_split(test_size=0.2, random_state=42)`

`LGBMRegressor` turns out as the best performing, with the minimum **sMAPE.**

`LGBMRegressor` **hyperparameters** after tuning with **Ray Tune** using Grid Search Algorithm:

- `boosting_type='gbdt'`
- `learning_rate=0.1`
- `max_bin=250`
- `max_depth=-1`
- `min_data_in_leaf=20`
- `num_iterations=100`
- `num_leaves=20`

***GPU** is leveraged.*

### <a href="https://github.com/georgemuriithi/shale-gas-wells/blob/main/Old-Wells-Prediction.ipynb">Old Wells</a>
<a href="https://colab.research.google.com/drive/1ytvFCquYvnic6fqAoLBGuLcIPTSMg3Eq?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

The following **exponential smoothing models** are tested:

- `SimpleExpSmoothing`
  - `smoothing_level=0.2`
  - `smoothing_level=0.6`
  - optimized smoothing level
- `Holt`
  - Additive model
  - Multiplicative model
  - Damped additive model
  - Damped multiplicative model
- `ExponentialSmoothing`
  - `use_boxcox=True`
    - Additive model
    - Damped additive model

Depending on the model with the minimum **SSE (Sum of Squared Error)** for each well, different models are used to forecast different wells.

### <a href="https://github.com/georgemuriithi/shale-gas-wells/blob/main/Investment-Decision.ipynb">Investment Decision</a>
<a href="https://colab.research.google.com/drive/1aFY-WH7U4QJpItl5yXOoEvJC0_t06hDn?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

The following 0-1 integer programming model is used:

<p align="center">
  <img src="https://user-images.githubusercontent.com/21691211/209959598-f4f64642-4fe1-4423-a421-824da4195b66.png">
</p>
