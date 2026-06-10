import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AutoDS",
    layout="wide"
)

st.title("🚀 AutoDS - Data Science Platform")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    file_content = uploaded_file.getvalue()

    if (
        "file_content" not in st.session_state
        or st.session_state.file_content != file_content
    ):

        st.session_state.file_content = file_content

        try:
            st.session_state.df = pd.read_csv(
                uploaded_file,
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            try:
                st.session_state.df = pd.read_csv(
                    uploaded_file,
                    encoding="latin1"
                )

            except Exception:
                st.error(
                    "Unable to read the file. Please upload a valid CSV."
                )
                st.stop()

    df = st.session_state.df

    # Reset Dataset
    if st.button("Reset Dataset"):
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset reset successfully!")
        st.rerun()

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Shape
    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # Column Names
    st.subheader("Columns")
    st.write(list(df.columns))

    # Data Types
    st.subheader("Data Types")

    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Data Type"]

    st.dataframe(dtype_df)

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(missing_df)

    missing_only = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if len(missing_only) > 0:

        st.subheader("Missing Values Chart")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(
            missing_only["Column"],
            missing_only["Missing Values"]
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        left, center, right = st.columns([2, 3, 2])
        
        with center:
            st.pyplot(fig)
    else:
        st.success("No missing values found!")

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Histogram
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) > 0:

        st.subheader("Histogram")

        hist_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(
            df[hist_column].dropna(),
            bins=20
        )
        plt.tight_layout()
        left, center, right = st.columns([2, 3, 2])
        
        with center:
            st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    if len(numeric_columns) >= 2:

        corr_matrix = df[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            ax=ax,
            annot_kws={"size": 6}
        )

        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        left, center, right = st.columns([2, 3, 2])
        with center:
            st.pyplot(fig)
    else:
        st.info(
            "Need at least 2 numeric columns."
        )

    # Categorical Analysis
    st.subheader("Categorical Data Analysis")

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if len(categorical_columns) > 0:

        selected_cat = st.selectbox(
            "Select Categorical Column",
            categorical_columns
        )

        value_counts = df[selected_cat].value_counts()

        st.dataframe(value_counts)

        fig, ax = plt.subplots(figsize=(4, 2))

        value_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_cat}"
        )

        left, center, right = st.columns([2, 3, 2])
        with center:
            st.pyplot(fig)

    else:
        st.info(
            "No categorical columns found."
        )

    # Data Cleaning
    st.subheader("Data Cleaning")

    duplicate_count = df.duplicated().sum()

    st.write(
        f"Duplicate Rows Found: {duplicate_count}"
    )

    if st.button("Remove Duplicates"):

        rows_before = df.shape[0]

        df = df.drop_duplicates()

        st.session_state.df = df

        rows_after = df.shape[0]

        removed = rows_before - rows_after

        st.success(
            f"{removed} duplicate rows removed!"
        )

        st.rerun()

    # Missing Value Treatment
    st.subheader("Handle Missing Values")

    missing_columns = df.columns[
        df.isnull().sum() > 0
    ]

    if len(missing_columns) > 0:

        missing_column = st.selectbox(
            "Select Column",
            missing_columns
        )

        method = st.selectbox(
            "Select Method",
            [
                "Mean",
                "Median",
                "Mode",
                "Drop Rows"
            ]
        )

        if st.button(
            "Apply Missing Value Treatment"
        ):

            if (
                method in ["Mean", "Median"]
                and not pd.api.types.is_numeric_dtype(
                    df[missing_column]
                )
            ):

                st.error(
                    "Mean and Median work only for numeric columns."
                )

            else:

                if method == "Mean":

                    df[missing_column] = (
                        df[missing_column]
                        .fillna(
                            df[missing_column].mean()
                        )
                    )

                elif method == "Median":

                    df[missing_column] = (
                        df[missing_column]
                        .fillna(
                            df[missing_column].median()
                        )
                    )

                elif method == "Mode":

                    df[missing_column] = (
                        df[missing_column]
                        .fillna(
                            df[missing_column].mode()[0]
                        )
                    )

                elif method == "Drop Rows":

                    df = df.dropna(
                        subset=[missing_column]
                    )

                st.session_state.df = df

                st.success(
                    f"Missing values handled using {method}"
                )

                st.dataframe(df.head())
                st.write("Remaining missing values:", df[missing_column].isnull().sum())

    else:
        st.success(
            "No missing values found."
        )

    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns for outlier detection")
    else:
        outlier_column = st.selectbox(
            "Select Column for Outlier Detection",
            numeric_cols
        )

    Q1 = df[outlier_column].quantile(0.25)
    Q3 = df[outlier_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[outlier_column] < lower_bound)
        |
        (df[outlier_column] > upper_bound)
    ]
    st.write(
        f"Number of Outliers: {len(outliers)}"
    )

    if len(outliers) > 0:
        st.dataframe(outliers.head())
    
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.boxplot(
        x=df[outlier_column].dropna(),
        ax=ax
    )
    ax.set_title(
        f"Outliers in {outlier_column}"
    )
    plt.tight_layout()
    left, center, right = st.columns([2, 3, 2])
    
    with center:
        st.pyplot(fig)

    if st.button("Remove Outliers"):
        df = df[
            (df[outlier_column] >= lower_bound)
            &
            (df[outlier_column] <= upper_bound)
        ]
        
        st.session_state.df = df
        st.success(
            f"{len(outliers)} outliers removed!"
        )
        st.rerun()
    
    # Machine Learning Preparation
    st.subheader("Machine Learning")
    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )
    feature_columns = st.multiselect(
        "Select Features",
        [col for col in df.columns if col != target_column]
    )
    model_type = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Random Forest",
            "Logistic Regression"
        ]
    )

    if st.button("Train Model"):
        if len(feature_columns) == 0:
            st.error("Please select at least one feature.")
        
        else:
            X = df[feature_columns].copy()
            y = df[target_column]
            # Encode categorical columns
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            if len(X.columns) == 0:
                st.error("Select numeric feature columns.")
                
            else:
                X = df[feature_columns]
                y = df[target_column]
                
                # Convert X if needed
                X = X.select_dtypes(include=np.number)
                
                # 🔥 BEFORE train-test split (this is what I meant)
                y = df[target_column]

                # Check missing values
                if X.isnull().sum().sum() > 0:
                    st.warning(
                        "⚠ Dataset contains missing values. Please use the Missing Value Treatment section before training."
                    )
                    st.stop()

                # Check infinite values
                if np.isinf(X.select_dtypes(include=np.number)).sum().sum() > 0:
                    st.warning(
                        "⚠ Dataset contains infinite values."
                    )
                    st.stop()

                # Check empty dataset
                if len(X) == 0:
                    st.warning(
                        "⚠ No rows available for training."
                    )
                    st.stop()
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
                
                if model_type == "Linear Regression":
                    if pd.api.types.is_numeric_dtype(y) is False:
                        st.error(
                            "Linear Regression is for continuous targets. Use Logistic Regression or Random Forest."
                        )
                
                    else:
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        st.write("Target unique values:", len(np.unique(y)))
                        st.write("Prediction sample:", predictions[:10])
                        st.write("Target sample:", y_test.head(10))

                        score = r2_score(
                            y_test,
                            predictions
                        )

                        st.success("Model trained successfully!")
                        st.metric(
                            "R² Score",
                            round(score, 4)
                        )
                    
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(
                        max_iter=1000
                    )
                    
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    st.write("Target unique values:", len(np.unique(y)))
                    st.write("Prediction sample:", predictions[:10])
                    st.write("Target sample:", y_test.head(10))

                    score = accuracy_score(
                        y_test,
                        predictions
                    )
                    
                    st.success("Model trained successfully!")
                    st.metric(
                        "Accuracy",
                        round(score, 4)
                    )
                
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    st.write("Target unique values:", len(np.unique(y)))
                    st.write("Prediction sample:", predictions[:10])
                    st.write("Target sample:", y_test.head(10))

                    score = accuracy_score(y_test, predictions)

                    st.success("Model trained successfully!")
                    st.metric("Accuracy", round(score, 4))


                    score = accuracy_score(
                        y_test,
                        predictions
                    )
            
                    st.success("Model trained successfully!")
                    st.metric(
                        "Accuracy",
                        round(score, 4)
                    )

    # Download Dataset
    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )