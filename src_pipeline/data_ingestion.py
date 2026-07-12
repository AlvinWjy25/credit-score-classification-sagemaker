import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import traceback
from IPython.display import display
from config import RAW_DATA_PATH, TARGET_COLUMN, PROCESSED_DATA_PATH, ARTIFACT_DIR

import warnings
warnings.filterwarnings("ignore") #debug

class CreditDataCleaning:
    def __init__(self, csv_path, target_column):
        self.file_path = csv_path
        self.target_column = target_column
        self.df = None
        self.outlier_bounds = {}

    def load_data(self):
        """Membaca dataset dari path"""
        self.df = pd.read_csv(self.file_path, low_memory=False)
        print(f"[data_ingestion.py]: Data is loaded successfully from '{self.file_path}'.")
        return self.df

    def filter_outlier(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[df[column] > upper_bound]
        return outliers, upper_bound

    def check_outlier(self, upper_bound, column):
        outliers = self.df[self.df[column] > upper_bound]
        print(f"[data_ingestion.py]: Jumlah outlier: {len(outliers)} | Upper Bound {column}: {upper_bound:.2f}")

    def cast_categorical_to_numeric(self):
        self.df["Age"] = pd.to_numeric(self.df["Age"], errors="coerce")
        self.df["Annual_Income"] = pd.to_numeric(self.df["Annual_Income"], errors="coerce")
        self.df["Num_of_Loan"] = pd.to_numeric(self.df["Num_of_Loan"], errors="coerce")
        self.df["Num_of_Delayed_Payment"] = self.df["Num_of_Delayed_Payment"].astype(str).str.replace(r"[^\d]", "", regex=True)
        self.df["Num_of_Delayed_Payment"] = pd.to_numeric(self.df["Num_of_Delayed_Payment"], errors="coerce")
        self.df["Changed_Credit_Limit"] = pd.to_numeric(self.df["Changed_Credit_Limit"], errors="coerce")
        self.df["Outstanding_Debt"] = pd.to_numeric(self.df["Outstanding_Debt"], errors="coerce")
        self.df["Amount_invested_monthly"] = pd.to_numeric(self.df["Amount_invested_monthly"], errors="coerce")

        return self.df

    def history_to_months(self, x):
        if pd.isna(x):
            return np.nan
        match = re.search(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(x))

        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years * 12 + months
        
        return np.nan

    def count_loans(self, x):
        if pd.isna(x):
            return 0
        loans = x.replace(" and ", ", ").split(",")
        return len([loan.strip() for loan in loans if loan.strip()])


    def fix_numerical_outliers(self, ver):
        self.outlier_bounds = {}

        # Fix Age
        self.df.loc[(self.df['Age'] < 18) | (self.df['Age'] > 100), 'Age'] = np.nan
        self.df['Age'] = self.df.groupby("Customer_ID")["Age"].ffill().bfill()
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

        # Fix Annual Income
        self.df.loc[self.df['Annual_Income'] > 300000, 'Annual_Income'] = np.nan
        _, upper_annual = self.filter_outlier(self.df, "Annual_Income")
        self.outlier_bounds["Annual_Income"] = upper_annual
        self.df.loc[self.df['Annual_Income'] > upper_annual, 'Annual_Income'] = np.nan
        self.df['Annual_Income'] = self.df.groupby('Occupation')['Annual_Income'].transform(lambda x: x.fillna(x.median()))

        # Fix Interest Rate
        _, upper_interest = self.filter_outlier(self.df, "Interest_Rate")
        self.outlier_bounds["Interest_Rate"] = upper_interest
        self.df.loc[self.df['Interest_Rate'] > upper_interest, 'Interest_Rate'] = np.nan
        self.df['Interest_Rate'] = self.df.groupby('Customer_ID')['Interest_Rate'].ffill().bfill()
        self.df['Interest_Rate'] = self.df['Interest_Rate'].fillna(self.df['Interest_Rate'].median())

        # Fix Monthly Inhand Salary (Gunakan upper_salary, BUKAN upper_interest!)
        self.df['Monthly_Inhand_Salary'] = self.df.groupby('Customer_ID')['Monthly_Inhand_Salary'].ffill().bfill()
        self.df['Monthly_Inhand_Salary'] = self.df.groupby('Occupation')['Monthly_Inhand_Salary'].transform(lambda x: x.fillna(x.median()))
        _, upper_salary = self.filter_outlier(self.df, "Monthly_Inhand_Salary")
        self.outlier_bounds["Monthly_Inhand_Salary"] = upper_salary
        self.df['Monthly_Inhand_Salary'] = self.df['Monthly_Inhand_Salary'].clip(upper=upper_salary)

        # Fix Number of Credit Card
        _, upper_credit = self.filter_outlier(self.df, "Num_Credit_Card")
        self.outlier_bounds["Num_Credit_Card"] = upper_credit
        self.df.loc[self.df['Num_Credit_Card'] > 12, 'Num_Credit_Card'] = np.nan
        self.df['Num_Credit_Card'] = self.df.groupby('Customer_ID')['Num_Credit_Card'].ffill().bfill()
        self.df['Num_Credit_Card'] = self.df['Num_Credit_Card'].fillna(self.df['Num_Credit_Card'].median())
        self.df['Num_Credit_Card'] = self.df['Num_Credit_Card'].clip(upper=11)

        # Fix Delay from Due Date
        self.df['Delay_from_due_date'] = self.df['Delay_from_due_date'].clip(lower=0)
        self.df['Delay_from_due_date'] = self.df.groupby('Customer_ID')['Delay_from_due_date'].ffill().bfill()
        self.df['Delay_from_due_date'] = self.df['Delay_from_due_date'].fillna(self.df['Delay_from_due_date'].median())

        # Fix Changed Credit Limit
        self.df['Changed_Credit_Limit'] = self.df.groupby('Customer_ID')['Changed_Credit_Limit'].ffill().bfill()
        self.df['Changed_Credit_Limit'] = self.df['Changed_Credit_Limit'].fillna(self.df['Changed_Credit_Limit'].median())
        self.df['Changed_Credit_Limit'] = self.df['Changed_Credit_Limit'].clip(lower=0)


        # Fix Number of Loans and Delayed Payment
        self.df['Num_of_Loan'] = self.df['Num_of_Loan'].clip(lower=0)
        self.df['Num_of_Delayed_Payment'] = self.df['Num_of_Delayed_Payment'].clip(lower=0)
        self.df['Num_of_Loan'] = self.df.groupby('Customer_ID')['Num_of_Loan'].ffill().bfill()
        self.df['Num_of_Delayed_Payment'] = self.df.groupby('Customer_ID')['Num_of_Delayed_Payment'].ffill().bfill()
        self.df['Num_of_Loan'] = self.df['Num_of_Loan'].fillna(0)
        self.df['Num_of_Delayed_Payment'] = self.df['Num_of_Delayed_Payment'].fillna(0)

        _, upper_bound_loan = self.filter_outlier(self.df, "Num_of_Loan")
        _, upper_bound_delayed = self.filter_outlier(self.df, "Num_of_Delayed_Payment")
        self.outlier_bounds["Num_of_Loan"] = upper_bound_loan
        self.outlier_bounds["Num_of_Delayed_Payment"] = upper_bound_delayed

        self.df['Num_of_Loan'] = self.df['Num_of_Loan'].clip(upper=upper_bound_loan)
        self.df['Num_of_Delayed_Payment'] = self.df['Num_of_Delayed_Payment'].clip(upper=upper_bound_delayed)

        # Fix Number of Credit Inquiries
        self.df['Num_Credit_Inquiries'] = self.df['Num_Credit_Inquiries'].clip(lower=0)
        self.df['Num_Credit_Inquiries'] = self.df.groupby('Customer_ID')['Num_Credit_Inquiries'].ffill().bfill()
        self.df['Num_Credit_Inquiries'] = self.df['Num_Credit_Inquiries'].fillna(0)
        _, upper_bound_inquiries = self.filter_outlier(self.df, "Num_Credit_Inquiries")
        self.outlier_bounds["Num_Credit_Inquiries"] = upper_bound_inquiries
        self.df['Num_Credit_Inquiries'] = self.df['Num_Credit_Inquiries'].clip(upper=upper_bound_inquiries)

        # Fix Outstanding Debt
        self.df["Outstanding_Debt"] = self.df["Outstanding_Debt"].clip(lower=0)
        self.df["Outstanding_Debt"] = self.df.groupby('Customer_ID')['Outstanding_Debt'].ffill().bfill()
        self.df["Outstanding_Debt"] = self.df["Outstanding_Debt"].fillna(0)

        # Fix Amount Invested Monthly
        self.df["Amount_invested_monthly"] = self.df["Amount_invested_monthly"].clip(lower=0)
        self.df["Amount_invested_monthly"] = self.df.groupby('Customer_ID')['Amount_invested_monthly'].ffill().bfill()
        self.df["Amount_invested_monthly"] = self.df["Amount_invested_monthly"].fillna(0)

        # Fix Monthly Balance
        self.df["Monthly_Balance"] = self.df["Monthly_Balance"].clip(lower=0)
        self.df["Monthly_Balance"] = self.df.groupby('Customer_ID')['Monthly_Balance'].ffill().bfill()
        self.df["Monthly_Balance"] = self.df["Monthly_Balance"].fillna(0)

        # Fix Credit History Age Months

        self.df["Credit_History_Age_Months"] = self.df["Credit_History_Age"].apply(self.history_to_months)
        self.df["Credit_History_Age_Months"] = self.df["Credit_History_Age_Months"].clip(lower=0)
        self.df["Credit_History_Age_Months"] = self.df.groupby('Customer_ID')['Credit_History_Age_Months'].ffill().bfill()
        self.df["Credit_History_Age_Months"] = self.df["Credit_History_Age_Months"].fillna(self.df["Credit_History_Age_Months"].median())

        # --- Verifikasi Bebas Outlier Ekstrem ---
        if(ver == 1):
            print("\n=== Check Outliers Post-Processing ===")
            for column, upper_b in self.outlier_bounds.items():
                self.check_outlier(upper_b, column)

            print("[data_ingestion.py]: Outlier handling is completed.")
        return self.df

   
    def handle_categorical_columns(self):
       
        # Fixing Missing Value in Categorical
        self.df["Occupation"] = self.df["Occupation"].replace("_______", "Unknown").fillna("Unknown")
        self.df["Credit_Mix"] = self.df["Credit_Mix"].replace("_", "Unknown").fillna("Unknown")
       
        # Feature Extraction dari Payment Behaviour
        self.df["Spending_Level"] = self.df["Payment_Behaviour"].str.extract(r"(High_spent|Low_spent)").fillna("Unknown")
        self.df["Payment_Size"] = self.df["Payment_Behaviour"].str.extract(r"(Small|Medium|Large)").fillna("Unknown")

        # MultiLabelBinarizer untuk Type_of_Loan (Tanpa loop duplikat lama)
        self.df["Loan_Count"] = self.df["Type_of_Loan"].apply(self.count_loans)
        loan_lists = self.df["Type_of_Loan"].fillna("").str.replace(" and ", ", ", regex=False).str.split(",")
        loan_lists = loan_lists.apply(lambda x: [i.strip() for i in x if i.strip()])
       
        mlb = MultiLabelBinarizer()
        loan_binary_array = mlb.fit_transform(loan_lists)
        columns_with_prefix = [f"Loan_{c.replace(' ', '_').replace('-', '_')}" for c in mlb.classes_]
        loan_df = pd.DataFrame(loan_binary_array, columns=columns_with_prefix, index=self.df.index).astype(np.int8)
        self.df = pd.concat([self.df, loan_df], axis=1)

        return self.df

    def finalize_dataframe(self):
        drop_cols = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Payment_Behaviour', 'Credit_History_Age', 'Loan_Unknown', 'loan_unknown']
        self.df = self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], errors='ignore')
        return self.df

    def final_check_df(self, ver = 0):
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        num_cols = self.df.select_dtypes(include=["float64", "int64", "int32", "int8"]).columns

        if (ver == 1):
            print("\n=== Final Check Dataframe ===")
            if len(cat_cols) > 0:
                display(self.df[cat_cols].describe())
            display(self.df[num_cols].describe())
            print("[data_ingestion.py]: Missing Values Summary:")
            print(self.df.isna().sum())

    def run_all_cleaner(self, ver):
        print("[data_ingestion.py]: Starting Preprocessing Pipeline...")
        self.load_data()
        self.cast_categorical_to_numeric()
        self.fix_numerical_outliers(ver)
        self.handle_categorical_columns()
        self.finalize_dataframe()
       
        self.final_check_df(ver)
        print("\n[data_ingestion.py]: Pipeline executed successfully! Data is ready for splitting/modeling.")

        return self.df

class UnifiedCreditPreprocessor(CreditDataCleaning):
    def __init__(self, csv_path, target_column):
        super().__init__(csv_path, target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def encode_categorical_features(self):
        # 1. Custom Ordinal Encoding
        credit_mix_map = {'Bad': 0, 'Unknown': 1, 'Standard': 2, 'Good': 3}
        spending_map = {'Low_spent': 0, 'Unknown': 1, 'High_spent': 2}
        payment_size_map = {'Small': 0, 'Unknown': 1, 'Medium': 2, 'Large': 3}

        self.df['Credit_Mix'] = self.df['Credit_Mix'].map(credit_mix_map).fillna(1)
        self.df['Spending_Level'] = self.df['Spending_Level'].map(spending_map).fillna(1)
        self.df['Payment_Size'] = self.df['Payment_Size'].map(payment_size_map).fillna(1)

        if 'Payment_of_Min_Amount' in self.df.columns:
            self.df['Payment_of_Min_Amount'] = self.df['Payment_of_Min_Amount'].str.strip().map({'Yes': 1, 'No': 0}).fillna(0.5)

        Month_map = {'January': 0, "Febuary": 1, "March": 2, "April": 3, "May": 4, "June": 5, "July": 6, "August": 7,
                     "September": 8, "October": 9, "November": 10, "December": 11}
        self.df['Month'] = self.df['Month'].map(Month_map).fillna(1)
       
        # 2. Custom One-Hot Encoding untuk Occupation
        if 'Occupation' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Occupation'], prefix='Occ', drop_first=True, dtype=int)

        # Target Variable Mapping
        if self.target_column in self.df.columns:
            score_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
            self.df[self.target_column] = self.df[self.target_column].map(score_map)
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n[data_ingestion.py]: Data split into training and testing sets with test size = {test_size}.")
        print(f"[data_ingestion.py]: X_train set: {self.X_train.shape} | X_test set: {self.X_test.shape}")

        self.X_train, self.X_test = self.normalize_data()
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def normalize_data(self):
        self.scaler = StandardScaler()

        print("[data_ingestion.py]: Normalisasi Data (StandardScaler)...")
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)
        print("[data_ingestion.py]: Success!")
        return self.X_train, self.X_test

    def export_data(self, output_dir):
        if self.X_train is None or self.y_train is None:
            print("[data_ingestion.py]: Error: Data belum di-split. Jalankan split_data() atau fit_transform() terlebih dahulu.")
            return
        
        artifact_dir = Path(ARTIFACT_DIR)
        joblib.dump(self.scaler, artifact_dir / "scaler.joblib")
        print("[data_ingestion.py]: saved to artifacts/scaler.joblib")

        self.X_train.to_csv(output_dir / "ingested" / "X_train.csv", index=False)
        self.X_test.to_csv(output_dir / "ingested" / "X_test.csv", index=False)
        self.y_train.to_csv(output_dir / "ingested" / "y_train.csv", index=False)
        self.y_test.to_csv(output_dir / "ingested" / "y_test.csv", index=False)
        print(f"\n[data_ingestion.py]: X_train, X_test, y_train, dan y_test berhasil diekspor ke '{output_dir}'")

    def fit(self, ingested_path, target_column, ver = 0):
        self.ingest_path = ingested_path
        self.df = self.run_all_cleaner(ver)
        return self.df

    def fit_transform(self, test_size=0.2, random_state=42):
        self.df = self.encode_categorical_features()
        self.split_data(test_size=test_size, random_state=random_state)
        self.export_data(self.ingest_path)

        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    
    print("--- [data_ingestion.py]: Menjalankan Test Lokal Preprocessing ---")
    try:
        # print(RAW_DATA_PATH)

        preprocessor = UnifiedCreditPreprocessor(RAW_DATA_PATH, TARGET_COLUMN)
        preprocessor.fit(PROCESSED_DATA_PATH, TARGET_COLUMN, ver=1)  
        
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(test_size=0.2)
        print("\n[data_ingestion.py]: Pipeline berhasil!")
        print(f"[data_ingestion.py]: Shape X_train: {X_train.shape}")
        print(f"[data_ingestion.py]: Shape X_test: {X_test.shape}")
        
    except Exception as e:
        # print(f"\n[TEST GAGAL] Terjadi error pada pipeline: {e}")
        print({e})
        print(traceback.print_exc())
