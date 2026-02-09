# author:- abhisheksoni


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



st.set_page_config(page_title="Stock ML Model Comparison", layout="wide")
st.title(" Stock Fundamental Analysis – Model Evaluation")


uploaded_file = st.file_uploader("Upload Stock CSV", type=["csv"])




def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k.lower() in col.lower():
                return col
    return None



if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())


 
    col_map = {
        "CMP": find_col(df, ["CMP"]),
        "PE": find_col(df, ["P/E"]),
        "MarketCap": find_col(df, ["Cap"]),
        "DivYield": find_col(df, ["Div"]),
        "NetProfit": find_col(df, ["NP"]),
        "ProfitGrowth": find_col(df, ["Profit Var"]),
        "Sales": find_col(df, ["Sales Qtr"]),
        "SalesGrowth": find_col(df, ["Sales Var"]),
        "ROCE": find_col(df, ["ROCE"])
    }


    if None in col_map.values():
        st.error("Required columns not found in CSV")
        st.stop()

    df = df.rename(columns={v: k for k, v in col_map.items()})


 
    num_cols = list(col_map.keys())

    for col in num_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=num_cols, inplace=True)

   

    st.sidebar.header("Company Selection Rules")


    roce = st.sidebar.slider("Min ROCE (%)", 0, 50, 20)
    sales_g = st.sidebar.slider("Min Sales Growth (%)", -100, 100, 10)
    profit_g = st.sidebar.slider("Min Profit Growth (%)", -100, 100, 10)
    pe = st.sidebar.slider("Max P/E", 0, 100, 40)
    mcap = st.sidebar.slider("Min Market Cap (Cr)", 0, 50000, 1000)


    df["Good_Stock"] = (
        (df["ROCE"] > roce) &
        (df["SalesGrowth"] > sales_g) &
        (df["ProfitGrowth"] > profit_g) &
        (df["PE"] < pe) &
        (df["MarketCap"] > mcap)
    ).astype(int)


    X = df[num_cols]
    y = df["Good_Stock"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    st.sidebar.header("Model Hyperparameters")


    log_c = st.sidebar.slider("Logistic Regression C", 0.01, 5.0, 1.0)


    rf_trees = st.sidebar.slider("RF Trees", 50, 500, 300)
    rf_depth = st.sidebar.slider("RF Max Depth", 2, 20, 10)


    xgb_trees = st.sidebar.slider("XGB Trees", 50, 500, 200)
    xgb_depth = st.sidebar.slider("XGB Max Depth", 2, 10, 5)
    xgb_lr = st.sidebar.slider("XGB Learning Rate", 0.01, 0.5, 0.1)



    log_model = LogisticRegression(C=log_c, max_iter=1000)
    log_model.fit(X_train, y_train)
    log_acc = accuracy_score(y_test, log_model.predict(X_test))


    rf_model = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=rf_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))


    xgb_model = XGBClassifier(
        n_estimators=xgb_trees,
        max_depth=xgb_depth,
        learning_rate=xgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

  

    st.subheader(" Model Accuracy Comparison")


    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression", f"{log_acc:.2f}")
    c2.metric("Random Forest", f"{rf_acc:.2f}")
    c3.metric("XGBoost", f"{xgb_acc:.2f}")



    best_model = max(
        [(log_acc, log_model), (rf_acc, rf_model), (xgb_acc, xgb_model)],
        key=lambda x: x[0]
    )[1]



    df["Prediction"] = best_model.predict(X)



    good_stocks = df[df["Prediction"] == 1]


    st.subheader(" Predicted Good Stocks (Best Model)")
    st.dataframe(good_stocks)


    st.download_button(
        " Download Results",
        good_stocks.to_csv(index=False),
        "predicted_good_stocks.csv"
    )
