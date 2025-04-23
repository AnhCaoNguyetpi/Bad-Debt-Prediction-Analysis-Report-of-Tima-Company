# Bad-Debt-Prediction-Analysis-Report-of-Tima-Company

# Analyzing Bank Loan Dataset: Risk and Repayment Behavior

This repository contains the code, report, and materials for the "Analyzing Bank Loan Dataset" project, which focuses on identifying risk factors, analyzing loan trends, and predicting bad debt and payment delay behavior using financial lending data.

## Project Background

The "Loan Risk and Repayment Analysis" project aims to analyze bank loan data to understand the key risk indicators, customer behavior, and repayment performance. By leveraging data analysis, data cleaning, visual dashboards (Power BI), and predictive modeling (Python), the project provides data-driven insights to improve lending strategies and reduce credit risk.

## Contents

- **Code**: Python scripts and Jupyter notebooks for data cleaning, preprocessing, EDA, model training, and prediction.
- **Report**: Summary of methodology, key findings, insights, and recommendations.
- **Datasets**: Original raw data and the cleaned dataset (Excel and CSV).
- **Figures**: Dashboards and visualizations generated using Power BI and Python.
- **References**: Tools and techniques used throughout the analysis.

## Project Phases

### 1. Data Collection
- **Total rows:** 2,383
- **Total columns:** 49
- **Tools used:** Excel (for initial review)

### 2. Data Preprocessing
- **Step 1:** Check and handle missing, duplicate, and invalid values.
- **Step 2:** Standardize and transform variables.
- **Step 3:** Verify data types and consistency.
- **Step 4:** Feature engineering to derive useful variables.
- **Step 5:** Final review and save the cleaned dataset.
- **Tools used:** Excel (raw data), Python (cleaning), Excel (output)

### 3. Exploratory Data Analysis & Dashboards
- **Tool:** Power BI
- **Dashboard Overview:**
  - Non-performing loan (NPL) ratio: 1%
  - Late payment ratio: 1%
  - Total customers: 1,541
  - On-time repayment rate: 75%
  - Total disbursed loans: 26.48 billion VND
  - Bad debt customers: 247 (16% of customers)
  - Popular loan types: phone and motorbike pawns, unsecured loans
  - Major occupations: Office workers
  - Key cities: Hanoi & Ho Chi Minh City

### 4. Trend & Customer Behavior Analysis
- Loans increased significantly from 2016–2018
- Males accounted for most loan applications
- Majority of customers have low to medium income
- High demand for liquid collateral (phones, motorbikes)

**Recommendations:**
- Focus on medium-high income groups with strong repayment capability
- Expand flexible loan products for personal asset pledges
- Review lending policy for low-income groups to widen market
- Maintain existing policies while enhancing for 2019
- Boost marketing and promotions to sustain loan growth

### 5. Risk Analysis
- Although current bad debt & delay rates are low (1%), credit risk potential is high (51%)
- High bad debt in groups with income < 10M VND and no income
- Even high-income groups (≥ 20M) show notable risk

**Recommendations:**
- Monitor high-risk profiles despite current low default
- Tighten loan criteria for low-income and risky occupations
- Improve risk evaluation process and post-loan support
- Apply guarantees or asset requirements for risky segments
- Set income thresholds for high-delay-risk industries
- Track debt trends during economic downturns

### 6. Predictive Modeling
- **Goal:** Build models to predict bad debt and late payment behavior
- **Tools:** Python, Scikit-learn
- **Approach:** Use features like income, job type, loan type, repayment history

## Usage

Explore the code, report, and dashboards in this repository. The analysis covers the full pipeline from data cleaning to risk prediction and actionable recommendations for credit policy improvement.

## Dependencies

This project relies on the following Python libraries and tools:

- **Pandas, NumPy** – Data manipulation and preprocessing
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Model training, evaluation
- **Power BI** – Dashboard creation and business insights
