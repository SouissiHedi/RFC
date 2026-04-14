import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Load data (same preprocessing contract)
# ----------------------------

def load():
    df = pd.read_csv("../data/ai4i2020.csv")

    df = df.rename(columns={
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear"
    })

    df['temp_diff'] = df['Process temperature'] - df['Air temperature']
    df['power'] = df['Rotational speed'] * df['Torque']  
    df['wear_per_rpm'] = df['Tool wear'] / (df['Rotational speed'] + 1e-6) 

    y = df["Machine failure"]
    X = df.drop(columns=["Machine failure", "UDI", "Product ID",
                        "TWF", "HDF", "PWF", "OSF", "RNF"])
    
    RANDOM_STATE = 42
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    test_indices = X_test.index

    print(df, X_train, X_test, y_train, y_test, test_indices, RANDOM_STATE)

    return df, X_train, X_test, y_train, y_test, test_indices, RANDOM_STATE

if __name__ == "__main__":
    load()