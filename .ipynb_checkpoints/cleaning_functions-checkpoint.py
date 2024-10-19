import pandas as pd

def standardize_column_names(df):
    df.columns = df.columns.str.replace('ST', 'state').str.lower().str.replace(' ', '_')
    return df

def clean_inconsistent_values(df):
    # Gender and State Mappings
    gender_mapping = {'F': 'F', 'M': 'M', 'Femal':'F', 'Male':'M', 'female':'F'}
    state_mapping = {'Washington': 'Washington', 'Arizona': 'Arizona', 'Nevada': 'Nevada', 
                     'California': 'California', 'Oregon': 'Oregon', 'Cali': 'California', 
                     'AZ': 'Arizona', 'WA': 'Washington'}
    
    # Standardize values in gender and state columns
    df['gender'] = df['gender'].map(gender_mapping)
    df['state'] = df['state'].map(state_mapping)
    df['education'] = df['education'].str.replace('Bachelors', 'Bachelor')
    
    # Ensure 'customer_lifetime_value' is string, replace '%', convert to float
    df['customer_lifetime_value'] = df['customer_lifetime_value'].astype(str).str.replace('%', '')
    df['customer_lifetime_value'] = pd.to_numeric(df['customer_lifetime_value'], errors='coerce')
    
    # Vehicle class mapping
    vehicle_class_mapping = {"Sports Car": "Luxury", "Luxury SUV": "Luxury", "Luxury Car": "Luxury"}
    df['vehicle_class'] = df['vehicle_class'].replace(vehicle_class_mapping)
    
    return df

import numpy as np

def correct_data_types(df):
    # Convert customer_lifetime_value to a float
    df['customer_lifetime_value'] = pd.to_numeric(df['customer_lifetime_value'], errors='coerce')

    # Ensure the number_of_open_complaints is correctly interpreted
    def extract_complaints(value):
        if isinstance(value, str):
            try:
                # Attempt to extract as integer from formatted strings
                return int(value.split('/')[1])
            except (IndexError, ValueError):
                return 0  # Default value if anything goes wrong
        elif isinstance(value, (int, float)):
            # Round up if not an integer float, handle else as integer
            return np.ceil(value) if not np.equal(value, np.floor(value)) else int(value)
        else:
            return 0  # Handle non-strings and NaNs as default of 0

    # Apply the function to correct the column
    df['number_of_open_complaints'] = df['number_of_open_complaints'].apply(extract_complaints)

    # Handle other numeric columns
    numerical_vars = ['customer_lifetime_value', 'income', 'monthly_premium_auto', 'total_claim_amount']
    for column in numerical_vars:
        if df[column].isna().sum() > 0:
            df[column] = df[column].fillna(df[column].median())
        df[column] = df[column].astype(int)

    return df

def handle_missing_values(df):
    # List of categorical columns
    categorical_vars = ['customer', 'state', 'gender', 'education', 'policy_type', 'vehicle_class']
    
    for column in categorical_vars:
        # Replace NaN with the moda of each categorical column
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df

def remove_duplicates_and_reset_index(df):
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def clean_customer_data(df):
    df = standardize_column_names(df)
    df = clean_inconsistent_values(df)
    df = correct_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates_and_reset_index(df)
    return df