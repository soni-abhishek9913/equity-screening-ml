# equity-screening-ml
# Stock Screening Engine

The stock market contains a large number of listed companies, making it difficult for investors to manually analyze and identify fundamentally strong stocks. Evaluating multiple financial indicators such as valuation, profitability, growth, and capital efficiency requires significant time and domain knowledge.

The objective of this project is to analyze stock fundamental data and build a system that allows users to define their own company selection criteria by adjusting key financial parameters, and then predict whether a company’s stock is good or not using machine learning techniques.

Financial Parameters Used

The analysis is performed using the following fundamental indicators:

S. No.

Company Name

CMP (Current Market Price)

P/E Ratio

Market Capitalization (Rs. Cr.)

Dividend Yield (%)

Quarterly Net Profit (Rs. Cr.)

Quarterly Profit Growth (%)

Quarterly Sales (Rs. Cr.)

Quarterly Sales Growth (%)

ROCE (Return on Capital Employed)

Page

Project Objective

The main goals of this project are:

To analyze stock fundamental data using key financial ratios

To allow users to create their own company selection logic by adjusting financial thresholds

To apply machine learning models to learn patterns from the data

To predict whether a company’s stock qualifies as a “Good Stock” or “Not Good Stock”

To assist investors and analysts in making data-driven screening decisions

Methodology Overview

Data Collection
Stock fundamental data is loaded from a CSV file.

Data Cleaning and Preprocessing
Missing values, inconsistent formats, and irrelevant data are handled automatically.

Parameter Adjustment
Users can dynamically adjust thresholds such as ROCE, P/E ratio, sales growth, and profit growth to define what they consider a good company.

Model Training and Evaluation
Machine learning models (Logistic Regression, Random Forest, and XGBoost) are trained and evaluated to classify stocks as good or not.

Prediction and Output
The best-performing model predicts high-quality stocks, and the results are displayed and exported for further analysis.

Expected Outcome

Automated identification of fundamentally strong stocks

Flexible company selection based on user-defined criteria

Improved efficiency in stock screening

Interactive and scalable stock analysis system

A machine learning–assisted engine for fundamental stock screening and evaluation.  
This project helps analyze stock fundamentals, apply custom business rules, compare multiple ML models, and identify high-quality stocks through an interactive Streamlit dashboard.




 Features

- Upload and analyze real-world stock fundamental data (CSV)
- Automatic column detection and data cleaning
- Customizable business rules for defining “good” stocks
- Machine Learning model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter tuning via UI
- Automatic best-model selection
- Downloadable list of selected stocks
- Interactive Streamlit dashboard



##  Machine Learning Approach

This project combines **rule-based labeling** with **supervised machine learning**:

1. User-defined financial thresholds create target labels
2. ML models learn patterns from fundamental indicators
3. Models are evaluated and compared dynamically
4. Best-performing model is used for final stock selection

> Note: This system is designed for **stock screening**, not direct future price prediction.


Input Features

- CMP (Current Market Price)
- P/E Ratio
- Market Capitalization
- Dividend Yield
- Quarterly Net Profit
- Quarterly Profit Growth
- Quarterly Sales
- Quarterly Sales Growth
- ROCE


Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- GitHub



 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py







Architecture Diagram 


┌─────────────────────┐
│   Stock CSV Upload  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Data Cleaning &     │
│ Column Mapping      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Engineering │
│ & Rule-Based Labels │
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────────────┐
│  Machine Learning Models            │
│  • Logistic Regression              │
│  • Random Forest                    │
│  • XGBoost                          │
└─────────┬──────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Model Evaluation &  │
│ Best Model Selection│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Stock Prediction &  │
│ Streamlit Dashboard │
└─────────────────────┘


