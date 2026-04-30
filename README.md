#  Portfolio Management using GRU RNNs and Multi-Objective Optimisation

> **Implementation of:** *"Deep Learning with Gated Recurrent Unit Recurrent Neural Networks and Multi-Objective Optimisation for Portfolio Management"*  
> Yang Liu & Lijun Yu — PRAI 2024 (IEEE)

---

##  Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Team](#team)

---

## Overview

This project implements and reproduces the paper's proposed framework for **stock return prediction** and **portfolio optimisation** using three models:

| Model | Type | Description |
|-------|------|-------------|
| **FFFFM** | Linear | Fama & French Five Factor Model — baseline regression |
| **v-SVR** | Machine Learning | Support Vector Regression with RBF kernel |
| **GRU RNNs** | Deep Learning | Gated Recurrent Unit Neural Networks |

The goal is to predict daily excess returns for **MSFT**, **AAPL**, and **SONY (SNE)** stocks, then use predicted returns with **multi-objective portfolio optimisation** (Pareto/Efficient Frontier) to recommend optimal asset weights that maximise return while minimising risk.

---

## Project Structure

```
portfolio-management-project-main/
│
├── data/
│   ├── portfolio_project/
│       ├── ffffm_predictions.csv       # Saved FFFFM model predictions
│       ├── gru_predictions.csv         # Saved GRU model predictions
│       ├── svr_predictions.csv         # Saved SVR model predictions
│       ├── train_data.csv              # Preprocessed training set (Jan 2000 – Dec 2011)
│       └── valid_data.csv              # Validation set (Dec 2011 – Nov 2013)
│   
│   
│     
│
├── AAPL.csv                            # Raw Apple stock price data
├── MSFT.csv                            # Raw Microsoft stock price data
├── SONY.csv                            # Raw Sony stock price data
├── F-F_Research_Data_5_Factors_2x3_daily...  # Fama-French factor data (raw)
├── FF5_daily.zip                       # Compressed Fama-French daily factors
├── data_collection.ipynb               # Data download, preprocessing & factor merging
│
├── models/
│   ├── fffm.ipynb                      # Fama-French Five Factor Model
│   ├── gru.ipynb                       # GRU Recurrent Neural Network
│   └── svr.ipynb                       # Support Vector Regression
│
├── optimization/
│   └── optimization.ipynb              # Multi-objective portfolio optimisation
│
├── results/                            # Output directory for results & figures
│
└── README.md
```

---

## Methodology

### 1. Data Collection & Preprocessing (`data_collection.ipynb`)
- Historical daily stock prices for **MSFT**, **AAPL**, **SONY** fetched from Yahoo Finance
- Five global Fama-French factors downloaded from the [French Data Library](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- Excess returns computed: `R_excess = R_stock - R_f` (risk-free rate)
- **Train split:** Jan 3, 2000 → Dec 2, 2011 (~2999 trading days)
- **Validation split:** Dec 5, 2011 → Nov 29, 2013 (~499 trading days)

### 2. Fama-French Five Factor Model (`fffm.ipynb`)
The five-factor linear regression equation:

```
R_it - R_Ft = a_i + b_i(R_Mt - R_Ft) + s_i·SMB + h_i·HML + r_i·RMW + c_i·CMA + e_it
```

| Factor | Meaning |
|--------|---------|
| **Mkt-RF** | Excess market return over risk-free rate |
| **SMB** | Small Minus Big (size premium) |
| **HML** | High Minus Low (value premium) |
| **RMW** | Robust Minus Weak (profitability) |
| **CMA** | Conservative Minus Aggressive (investment) |

- Trained using **Ordinary Least Squares (OLS)**
- Validated with **20-fold cross-validation** and p-value significance test

### 3. Support Vector Regression (`svr.ipynb`)
- Kernel: **Gaussian RBF** — `k(x1, x2) = exp(-λ||x-y||²)`
- Parameters calibrated via **grid search**: `C=2, v=0.5, λ=0.001`
- Uses same 5 factors as input; predicts excess return per stock

### 4. GRU Recurrent Neural Networks (`gru.ipynb`)
Architecture:
- **Input:** 5 factors, time step = 1
- **Hidden layers:** 2 × GRU layers, 400 neurons each, dropout = 0.2
- **Output:** Dense layer with 1 neuron (next-day return)
- Framework: **Keras + TensorFlow**

GRU equations:
```
z_t = σ(W_z · [h_{t-1}, x_t])          # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])          # Reset gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])   # Candidate activation
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Output state
```

### 5. Portfolio Optimisation (`optimization.ipynb`)
Multi-objective optimisation framework:

```
Minimise:   f1 = Σ Σ w_i · w_j · σ_ij    (Portfolio Risk)
Maximise:   f2 = Σ μ_i                    (Expected Return)
Subject to: Σ w_i = 1,  0 ≤ w_i ≤ 1
```

- **Efficient/Pareto Frontier** generated via 10,000 random portfolio samples
- Optimal weights selected from frontier (highest Sharpe-like trade-off)
- Final 5-day portfolio returns computed using predicted returns from all 3 models

---

## Results

### Dataset Visualisation

**Fig 4: Stock Price Data (MSFT, AAPL, SNE) — Jan 2000 to Nov 2013**

![Stock Prices](fig4_stock_prices.png)

**Fig 5: Five Fama-French Factors — First 50 Trading Days**

![Five Factors](fig5_five_factors.png)

---

### Model Performance (Validation R²)

| Model | R² (MSFT) | R² (AAPL) | R² (SONY) | p-value |
|-------|-----------|-----------|-----------|---------|
| **FFFFM** | 0.285 | 0.225 | 0.208 | p < 0.05  |
| **v-SVR** | 0.28 | 0.210 | 0.21 | p < 0.05  |
| **GRU RNNs** | **0.28** | **0.232** | **0.22** | p < 0.05 |

> All models are statistically consistent (p < 0.05 via 20-fold cross-validation).  
> **GRU RNNs achieves the best R² on AAPL and SONY**, confirming nonlinear models capture structure that linear regression misses.

---

### Portfolio Optimisation

**Fig 7: Efficient Frontier and Selected Optimal Portfolio**

![Efficient Frontier](fig7_efficient_frontier.png)

**Optimal Weights (from our implementation):**
- w1 (MSFT) = 0.01869
- w2 (AAPL) = 0.98017
- w3 (SONY) = 0.00114

**5-Day Total Returns Using Optimal Weights:**

| Day | v-SVR | GRU RNNs | FFFFM | Actual |
|-----|-------|----------|-------|--------|
| 1 | -0.04295 | -0.10684 | -0.076061 | -0.39796 |
| 2 | -0.18312 | -0.5105 | -0.312752 | -0.38414 |
| 3 | -2.0721 | -1.9947 | -1.99661 | -0.014031 |
| 4 | **1.8890** | **1.6059** | **2.037477** | 0.90063 |
| 5 | -1.57028 | -1.3662 | -1.618670 | -0.556557 |

---

## Setup & Installation

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn scipy
pip install tensorflow keras
pip install yfinance pandas-datareader
pip install jupyter
```

### Clone the Repository
```bash
git clone https://github.com/<your-username>/portfolio-management-project.git
cd portfolio-management-project
```

---

## How to Run

Run the notebooks **in order**:

```bash
# Step 1 — Download and preprocess data
jupyter notebook data_collection.ipynb

# Step 2 — Train and evaluate FFFFM (baseline)
jupyter notebook models/fffm.ipynb

# Step 3 — Train and evaluate SVR
jupyter notebook models/svr.ipynb

# Step 4 — Train and evaluate GRU RNN
jupyter notebook models/gru.ipynb

# Step 5 — Run portfolio optimisation
jupyter notebook optimization/optimization.ipynb
```

> **Google Colab users:** Each notebook auto-detects the Colab environment and mounts Google Drive. Just run all cells top to bottom.
You will also need to provide the code access to Google Drive to run it in collab

---

## References

1. Yang Liu & Lijun Yu, *"Deep Learning with GRU RNNs and Multi-Objective Optimisation for Portfolio Management,"* PRAI 2024, IEEE.
2. E. F. Fama & K. R. French, *"A Five-Factor Asset Pricing Model,"* Journal of Financial Economics, 2015.
3. K. Cho et al., *"On the properties of neural machine translation: Encoder-decoder approaches,"* SSST-8, 2014.
4. B. Schölkopf et al., *"New Support Vector Algorithms,"* Neural Computation, 2000.

---

## Team

| Name | Roll Number | Contribution |
|------|-------------|--------------|
| Ruthvik Devarakonda | 24AI10023 | Data Collection & Preprocessing |
| Banala Sai Chandan | 24AI10011 | Fama-French Five Factor Model (FFFFM) |
| Srishwan Katta | 24AI10030 | Support Vector Regression (SVR) |
| Sai Dharanidhar Ram | 24AI10004 | GRU Recurrent Neural Network |
| Aaron Jason Baptist | 24AI10015 | Portfolio Optimisation |

> *Project submitted as part of the Deep Learning course, IIT Kharagpur — Semester 4*
