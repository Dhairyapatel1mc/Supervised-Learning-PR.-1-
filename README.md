# 🏠 House Price Prediction — Regression & Gradient Descent

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=220&section=header&text=House%20Price%20Prediction&fontSize=48&fontAlignY=38&desc=Regression%20Models%20%7C%20Gradient%20Descent%20%7C%20Bias%E2%80%93Variance&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>
</p>

<p align="center">
  <b>End-to-End Machine Learning Regression Project</b><br>
  Dataset Understanding • EDA • Regression • Optimization • Model Diagnostics
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

This project develops and compares regression models for predicting house prices in INR. It covers the complete machine-learning workflow from dataset understanding and visualization to optimization and bias–variance diagnostics.

```text
Dataset
   ↓
Data Understanding & Preparation
   ↓
Exploratory Data Analysis
   ↓
Train / Test Split
   ↓
Simple Linear Regression
   ↓
Model Evaluation
   ↓
Multiple Linear Regression
   ↓
Polynomial Regression
   ↓
Batch GD / SGD / Mini-Batch GD
   ↓
Bias–Variance Analysis
   ↓
Model Diagnostics & Comparison
```

The project is designed to demonstrate both **practical model building** and the **mathematical intuition behind regression and gradient-based optimization**.

---

## 🎯 Objectives

- Identify independent and dependent variables.
- Understand and prepare the housing dataset.
- Visualize relationships between features and house price.
- Build Simple Linear Regression.
- Evaluate models using MSE, MAE, RMSE, R² and Adjusted R².
- Build Multiple Linear Regression using relevant features.
- Build Polynomial Regression of degree 2 and 3.
- Implement Batch Gradient Descent from scratch.
- Implement Stochastic Gradient Descent (SGD).
- Implement Mini-Batch Gradient Descent.
- Compare convergence behavior and training time.
- Analyze bias and variance.
- Identify underfitting and overfitting.
- Select a model based on generalization performance rather than training score alone.

---

## 📂 Dataset

The dataset contains **4,200 records** and **12 columns**.

### Dataset Schema

| Feature | Type | Missing Values | Role | Description |
|---|---|---:|---|---|
| `house_id` | `int64` | 0 | Feature | Dataset feature |
| `area_sqft` | `int64` | 0 | Feature | House area in square feet |
| `bedrooms` | `int64` | 0 | Feature | Number of bedrooms |
| `bathrooms` | `int64` | 0 | Feature | Number of bathrooms |
| `location_score` | `float64` | 0 | Feature | Numerical location quality score |
| `age_years` | `int64` | 0 | Feature | Age of the property in years |
| `distance_city_km` | `float64` | 0 | Feature | Distance from the city |
| `lot_size_sqft` | `int64` | 0 | Feature | Plot/lot size in square feet |
| `has_garage` | `int64` | 0 | Feature | Whether the property has a garage |
| `has_pool` | `int64` | 0 | Feature | Whether the property has a pool |
| `renovation_years_ago` | `int64` | 0 | Feature | Years since the property was renovated |
| `house_price_inr` | `int64` | 0 | Target | House price in Indian Rupees (target) |

> **Target variable:** `house_price_inr`

### Problem Type

**Supervised Learning → Regression**

The objective is to learn a function that maps house characteristics to a continuous house-price value.

---

## 🧠 Mathematical Foundation

For Simple Linear Regression:

```text
ŷ = β₀ + β₁x
```

For Multiple Linear Regression:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

The model parameters are learned by minimizing prediction error.

For Mean Squared Error:

```text
MSE = (1/n) × Σ(yᵢ − ŷᵢ)²
```

---

# 🔬 Part B — Dataset Understanding & Preparation

## Independent Variables

The predictive features include house characteristics such as:

- `area_sqft`
- `bedrooms`
- `bathrooms`
- `location_score`
- `age_years`
- `distance_city_km`
- `lot_size_sqft`
- `has_garage`
- `has_pool`
- `renovation_years_ago`

## Dependent Variable

```text
house_price_inr
```

This is the target variable that the models predict.

---

# 📊 Exploratory Data Analysis

The project visualizes relationships between important predictors and house price.

### Key visualizations

- House Area vs House Price
- Bedrooms vs House Price
- Bathrooms vs House Price
- Location Score vs House Price
- Property Age vs House Price
- Actual vs Predicted values
- Residual plots
- Gradient Descent convergence

Example:

```python
plt.scatter(df["area_sqft"], df["house_price_inr"])
plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price (INR)")
plt.title("House Area vs House Price")
plt.show()
```

### Why EDA?

EDA helps identify:

- Strong and weak relationships.
- Approximate linearity.
- Nonlinear patterns.
- Potential outliers.
- Features that may contribute useful predictive information.

---

# ✏️ Part C — Simple Linear Regression

Simple Linear Regression uses **one predictor**:

```text
Feature: area_sqft
Target : house_price_inr
```

The model learns:

```text
House Price = Intercept + Slope × Area
```

The slope represents the expected change in predicted price for a one-unit increase in house area within the fitted model.

A regression line is plotted against the observed test data to visually inspect the fit.

---

# 📏 Part D — Model Evaluation

The following metrics are used.

### MSE — Mean Squared Error

Measures the average squared prediction error.

**Lower is better.**

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Lower is better.**

### RMSE — Root Mean Squared Error

The square root of MSE.

```text
RMSE = √MSE
```

It is expressed in the same units as the target.

**Lower is better.**

### R² Score

Measures the proportion of target variation explained by the model.

**Higher is better.**

### Adjusted R²

Adjusted R² accounts for the number of predictors. It is useful when comparing models with different numbers of features.

---

# 📈 Part E — Multiple Linear Regression

Multiple Linear Regression uses several relevant features simultaneously.

```text
area_sqft
bedrooms
bathrooms
location_score
age_years
distance_city_km
lot_size_sqft
renovation_years_ago
        ↓
Multiple Linear Regression
        ↓
house_price_inr
```

This approach is more expressive than using house area alone because property prices are influenced by multiple characteristics.

### Why performance can improve

Adding relevant features can reduce bias because the model has more information about the target.

However, adding irrelevant or highly correlated features can increase variance and hurt generalization.

---

# 🔵 Part F — Polynomial Regression

Polynomial Regression extends a linear model with powers of the predictor.

### Degree 2

```text
ŷ = β₀ + β₁x + β₂x²
```

### Degree 3

```text
ŷ = β₀ + β₁x + β₂x² + β₃x³
```

Polynomial regression can model curved relationships that ordinary linear regression cannot represent.

The project compares linear and polynomial models both **visually and numerically**.

---

# ⚙️ Part G — Gradient Descent Optimization

Gradient Descent is an iterative optimization algorithm used to minimize a loss function.

General update rule:

```text
parameter_new = parameter_old − learning_rate × gradient
```

### 1. Batch Gradient Descent

Uses the entire training dataset for each update.

```text
All training samples
        ↓
Calculate gradient
        ↓
Update parameters
        ↓
Repeat
```

**Characteristics**
- Stable convergence.
- Smooth loss curve.
- More computation per update.

### 2. Stochastic Gradient Descent

Uses one sample at a time.

```text
Sample → Update
Sample → Update
Sample → Update
...
```

**Characteristics**
- Frequent updates.
- Potentially faster updates.
- Noisier convergence.

### 3. Mini-Batch Gradient Descent

Uses a small group of samples per update.

```text
Mini-batch → Update
Mini-batch → Update
Mini-batch → Update
```

**Characteristics**
- Good balance between Batch GD and SGD.
- Efficient computation.
- More stable than pure SGD.

### Convergence comparison

The loss curves are plotted to compare how quickly and smoothly each method approaches a low-error solution.

---

# ⚖️ Part H — Bias–Variance & Model Diagnostics

## Bias

High bias occurs when a model is too simple.

```text
High Bias
   ↓
Underfitting
```

The model fails to capture important patterns.

## Variance

High variance occurs when a model is too sensitive to its training data.

```text
High Variance
   ↓
Overfitting
```

The model performs very well on training data but substantially worse on unseen data.

---

## Model Complexity

```text
Simple Linear Regression
          ↓
   Low Complexity
          ↓
   Higher Bias Risk

Multiple Linear Regression
          ↓
   Moderate Complexity
          ↓
   Potentially Better Balance

Polynomial Regression
          ↓
   Higher Complexity
          ↓
   Higher Variance Risk
```

The optimal model is not necessarily the most complex model.

---

# 🚨 Underfitting vs Overfitting

### Underfitting

```text
Training Performance → Low
Testing Performance  → Low
```

The model is too simple.

### Good Generalization

```text
Training Performance → High
Testing Performance  → High
Train-Test Gap       → Small
```

### Overfitting

```text
Training Performance → Very High
Testing Performance  → Much Lower
Train-Test Gap       → Large
```

---

# 📊 Model Results

The current project analysis gives approximately:

| Model | Test R² | Adjusted R² | Test RMSE |
|---|---:|---:|---:|
| Simple Linear Regression | 0.5625 | 0.5620 | ₹8.18M |
| Polynomial Regression — Degree 2 | 0.5627 | — | ₹8.18M |
| Polynomial Regression — Degree 3 | 0.5629 | — | ₹8.18M |
| **Multiple Linear Regression** | **0.9178** | **0.9168** | **₹3.55M** |

> Results should be regenerated from the final notebook after any changes to preprocessing, feature selection, random seed, or train/test split.

### Main observation

Multiple Linear Regression substantially outperforms the single-feature models.

This indicates that **house area alone does not contain enough information to accurately explain house-price variation**, while multiple relevant property characteristics provide substantially more predictive information.

The small improvement from polynomial degrees 2 and 3 also suggests that simply adding higher-order powers of `area_sqft` is less useful than incorporating additional relevant features.

---

# 🏆 Model Selection Strategy

The preferred model should be selected using **test/generalization performance**, not training performance alone.

A strong candidate should have:

```text
High Test R²
      +
Low Test MSE / MAE / RMSE
      +
Small Train-Test Performance Gap
```

Based on the current results, **Multiple Linear Regression is the strongest model among the tested approaches**.

---

# 📸 Visualizations

Store generated plots inside the `images/` folder and reference them here:

```markdown
## 📊 Exploratory Data Analysis

![Feature Relationships](images/feature_relationships.png)

## 📈 Regression Comparison

![Regression Comparison](images/regression_comparison.png)

## ⚙️ Gradient Descent Convergence

![Gradient Descent](images/gradient_descent_convergence.png)

## 🔍 Residual Diagnostics

![Residual Plot](images/residual_plot.png)

## 📊 Actual vs Predicted

![Actual vs Predicted](images/actual_vs_predicted.png)
```

---

# 🗂️ Recommended Project Structure

```text
House-Price-Prediction/
│
├── data/
│   └── house_price_dataset.csv
│
├── notebooks/
│   └── house_price_regression.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── regression_models.py
│   ├── gradient_descent.py
│   └── evaluation.py
│
├── images/
│   ├── feature_relationships.png
│   ├── regression_comparison.png
│   ├── gradient_descent_convergence.png
│   ├── residual_plot.png
│   └── actual_vs_predicted.png
│
├── results/
│   └── model_results.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Regression, preprocessing and metrics |
| Jupyter Notebook | Experimentation and documentation |

---

# 🚀 Installation

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter:

```bash
jupyter notebook
```

---

# 📦 requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# ▶️ How to Run

1. Put the CSV dataset in `data/`.
2. Open the Jupyter notebook.
3. Run data-understanding and EDA cells.
4. Create the train/test split.
5. Train Simple Linear Regression.
6. Calculate evaluation metrics.
7. Train Multiple Linear Regression.
8. Train Polynomial Regression.
9. Run the three Gradient Descent implementations.
10. Compare convergence and model performance.
11. Analyze residuals and train/test performance.
12. Identify the best bias–variance balance.

---

# 💡 Future Improvements

Possible next steps for a production-oriented version:

- Cross-validation.
- Feature scaling pipelines.
- Feature selection.
- Ridge and Lasso Regression.
- Hyperparameter tuning.
- Outlier treatment.
- Residual normality analysis.
- Heteroscedasticity testing.
- Learning curves.
- Random Forest Regression.
- Gradient Boosting / XGBoost.
- SHAP-based model explainability.
- FastAPI model serving.
- Streamlit prediction dashboard.
- Model serialization and deployment.
- Experiment tracking.

---

# 🎓 Learning Outcomes

This project demonstrates practical understanding of:

```text
✓ Supervised Learning
✓ Regression
✓ Exploratory Data Analysis
✓ Feature / Target Selection
✓ Train-Test Splitting
✓ Simple Linear Regression
✓ Multiple Linear Regression
✓ Polynomial Regression
✓ MSE / MAE / RMSE
✓ R² / Adjusted R²
✓ Gradient Descent
✓ Batch Gradient Descent
✓ Stochastic Gradient Descent
✓ Mini-Batch Gradient Descent
✓ Bias–Variance Tradeoff
✓ Underfitting
✓ Overfitting
✓ Residual Diagnostics
✓ Model Selection
```

---

# 👨‍💻 Author

**Your Name**

AI/ML Learner • Python • Machine Learning • Data Science

- GitHub: `https://github.com/your-username`
- LinkedIn: `https://linkedin.com/in/your-profile`

---

<p align="center">
  <b>⭐ If this project helped you learn, consider giving the repository a star.</b>
</p>

<p align="center">
  Built with Python & Machine Learning 🚀
</p>
