"""
Titanic Survival Analysis Pipeline
Author: Meet Ahalpara

This script performs a comprehensive exploratory data analysis (EDA) on the historical 
Titanic passenger manifest. It is designed to be accessible to both technical and 
non-technical audiences, illustrating key demographic patterns that influenced survival rates.

Key Pipeline Stages:
1. Data Ingestion: Loading and structuring the raw passenger records.
2. Data Cleaning: Intelligently imputing missing age and embarkation data.
3. Feature Scaling: Normalizing financial metrics (fares) for balanced statistical comparison.
4. Visual Analytics: Generating intuitive, presentation-ready demographic dashboards.
"""

# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# PHASE 1: DATA INGESTION
# ==========================================
# The dataset file is dynamically located relative to this script's location.
# This ensures the script runs smoothly across different operating systems 
# without requiring the user to manually configure file paths.
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "Titanic-Dataset.csv")
df = pd.read_csv(csv_path)

print("\n" + "="*55)
print(" TITANIC SURVIVAL ANALYSIS PIPELINE INITIATED")
print("="*55)
print("\n[INFO] Loading historical passenger dataset...")
print("[SUCCESS] Dataset successfully loaded into memory.")
print(f"|-- Total Passengers (Rows): {df.shape[0]}")
print(f"|-- Total Attributes (Columns): {df.shape[1]}")
print("\n[PREVIEW] Initial Data Structure (First 5 Records):")
print("-" * 55)
print(df.head())
print("-" * 55)


# ==========================================
# PHASE 2: DATA CLEANING & IMPUTATION
# ==========================================
# Real-world data is rarely perfect. To maintain the integrity of the analysis, 
# missing passenger details must be addressed rather than discarding valuable records.

print("\n[INFO] Scanning dataset for missing values...")
print(df[['Age', 'Embarked']].isnull().sum().to_string())

# Impute (fill in) missing Age values.
# The median age is calculated for males and females independently, as gender 
# historically influenced passenger demographics, providing a more accurate estimation.
male_median_age = df[df['Sex'] == 'male']['Age'].median()
female_median_age = df[df['Sex'] == 'female']['Age'].median()

df.loc[(df['Age'].isnull()) & (df['Sex'] == 'male'), 'Age'] = male_median_age
df.loc[(df['Age'].isnull()) & (df['Sex'] == 'female'), 'Age'] = female_median_age

# Impute missing Embarkation ports using the most frequent boarding location (the statistical mode).
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

print("\n[SUCCESS] Missing values have been successfully resolved:")
print(df[['Age', 'Embarked']].isnull().sum().to_string())


# ==========================================
# PHASE 3: FEATURE SCALING (NORMALIZATION)
# ==========================================
# Ticket prices (fares) vary drastically. These values are scaled to a standard mathematical 
# range between 0.0 and 1.0. This normalizes the data, ensuring that extremely high 
# ticket prices do not disproportionately skew visual or statistical models.
scaler = MinMaxScaler()
print("\n[INFO] Normalizing financial metrics (Ticket Fares)...")
print(f"|-- Original Fare Range: ${df['Fare'].min():.2f} to ${df['Fare'].max():.2f}")

df['fare_normalized'] = scaler.fit_transform(df[['Fare']])

print("\n[PREVIEW] Normalized Fare Transformation (Sample View):")
print("-" * 55)
print(df[['Fare', 'fare_normalized']].head())
print("-" * 55)


# ==========================================
# PHASE 4: VISUAL ANALYTICS
# ==========================================
# A multi-panel visual dashboard is constructed to translate raw numbers into 
# clear, accessible insights regarding passenger survival trajectories.
print("\n[INFO] Generating visual demographic dashboards...")

plt.style.use('default')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Titanic Survival Patterns', fontsize=18, fontweight='bold', y=0.98)


# 4.1 Survival Rate by Passenger Class
# Visualizes the socio-economic divide in survival outcomes across 1st, 2nd, and 3rd classes.
sns.barplot(data=df, x='Pclass', y='Survived',
            hue='Pclass', palette='viridis',
            legend=False, dodge=False, ax=axes[0, 0])

axes[0, 0].set_title('Survival Rate by Passenger Class', fontweight='bold')
axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].grid(axis='y', alpha=0.3)


# 4.2 Survival Rate by Gender
# Highlights the demographic disparity driven by the historical "women and children first" protocol.
sns.barplot(data=df, x='Sex', y='Survived',
            hue='Sex', palette='coolwarm',
            legend=False, dodge=False, ax=axes[0, 1])

axes[0, 1].set_title('Survival Rate by Gender', fontweight='bold')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].grid(axis='y', alpha=0.3)


# 4.3 Survival Rate by Age Group
# Continuous age data is segmented into distinct generational brackets for clearer trend analysis.
df['age_group'] = pd.cut(
    df['Age'],
    bins=[0, 10, 20, 30, 40, 50, 60, 80],
    labels=['0–10', '10–20', '20–30', '30–40', '40–50', '50–60', '60+']
)

sns.barplot(data=df, x='age_group', y='Survived',
            hue='age_group', palette='plasma',
            legend=False, dodge=False, ax=axes[1, 0])

axes[1, 0].set_title('Survival Rate by Age Group', fontweight='bold')
axes[1, 0].set_ylabel('Survival Rate')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)


# 4.4 Survival Rate by Fare Quartile
# The passenger base is divided into four equal-sized financial groups to evaluate purchasing power.
df['fare_quartile'] = pd.qcut(
    df['fare_normalized'],
    4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

sns.barplot(data=df, x='fare_quartile', y='Survived',
            hue='fare_quartile', palette='magma',
            legend=False, dodge=False, ax=axes[1, 1])

axes[1, 1].set_title('Survival Rate by Fare Quartile', fontweight='bold')
axes[1, 1].set_ylabel('Survival Rate')
axes[1, 1].grid(axis='y', alpha=0.3)


# Layout Adjustment and Save Figure
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12,
                    wspace=0.25, hspace=0.35)

plt.savefig('titanic_demographic_survival_dashboard.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white')

plt.show()


# Cleanup Temporary Columns
df.drop(['age_group', 'fare_quartile'], axis=1, inplace=True)

print("\n[SUCCESS] Visual dashboard exported locally as 'titanic_demographic_survival_dashboard.png'.")
print("[INFO] Analytical data processing complete.")


# ==========================================
# PHASE 5: EXECUTIVE SUMMARY
# ==========================================
# Provides a concise, professional synthesis of the analytical findings.
print("\n" + "="*55)
print(" EXECUTIVE SUMMARY & KEY ANALYTICAL INSIGHTS")
print("="*55)

summary_report = """
1. Socio-Economic Stratification: Passenger class (Pclass) served as a critical 
   determinant of survival. First-class ticket holders exhibited significantly 
   higher survival probabilities than those in second or third class.
   
2. The Gender Delta: The dataset strongly reflects the historical "women and 
   children first" maritime protocol. Female passengers achieved substantially 
   higher survival rates across all demographic segments compared to men.
   
3. Generational Trends: Younger demographics, notably children and young adolescents, 
   demonstrated highly favorable survival trajectories.
   
4. Financial Correlation: Higher fare quartiles (indicating premium ticket purchases) 
   directly correlated with increased survival likelihoods, reinforcing the profound 
   impact of socio-economic status during the disaster.
"""

print(summary_report)
print("="*55 + "\n")
