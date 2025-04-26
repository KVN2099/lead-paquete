# Suppress all warnings and stderr output first
import warnings
import sys
import os

# Completely silence warnings
warnings.filterwarnings('ignore')

# Create a class to silence stderr
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Import standard libraries first
with SuppressStderr():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

# Import from framework with all errors suppressed
FULL_IMPORTS_AVAILABLE = False

# Try the basic import first
with SuppressStderr():
    try:
        from framework_Final import AnalisisDatosExploratorio
        
        # Try the additional imports
        try:
            from framework_Final import (
                NoSupervisados,
                Supervisado,
                evaluador,
                SerieTiempo,
                AnalisisDatosClasificacion
            )
            FULL_IMPORTS_AVAILABLE = True
        except:
            pass
            
    except Exception:
        pass

# Import required sklearn modules directly
with SuppressStderr():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_squared_error
    except:
        pass

# Check for optional packages without causing terminal errors
SKLEARN_EXTRA_AVAILABLE = False
UMAP_AVAILABLE = False

with SuppressStderr():
    try:
        # Just try to import the module name without accessing its contents
        import importlib.util
        if importlib.util.find_spec("sklearn_extra") is not None:
            SKLEARN_EXTRA_AVAILABLE = True
    except:
        pass

    try:
        if importlib.util.find_spec("umap") is not None:
            UMAP_AVAILABLE = True
    except:
        pass

# Set page configuration
st.set_page_config(page_title="Data Analysis Framework", layout="wide")

# Main title
st.title("Data Analysis and Model Training")

# Sidebar for navigation
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Exploratory Data Analysis", 
     "Unsupervised Learning", 
     "Supervised Learning (Regression)", 
     "Supervised Learning (Classification)",
     "Time Series Analysis"]
)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Delimiter selection
delimiter_options = {
    "Comma (,)": ",",
    "Semicolon (;)": ";"
}
delimiter = st.sidebar.selectbox(
    "Select delimiter",
    list(delimiter_options.keys())
)

# Decimal separator
decimal_options = {
    "Period (.)": ".",
    "Comma (,)": ","
}
decimal = st.sidebar.selectbox(
    "Select decimal separator",
    list(decimal_options.keys())
)

# Process data when file is uploaded
if uploaded_file is not None:
    # Map selected options to actual values
    delimiter_value = delimiter_options[delimiter]
    decimal_value = decimal_options[decimal]
    
    # Check if we can proceed based on the selected analysis type
    if not FULL_IMPORTS_AVAILABLE and analysis_type in ["Unsupervised Learning", "Supervised Learning (Classification)"]:
        st.warning(f"Limited functionality mode: Some required packages couldn't be loaded for {analysis_type}. Try selecting a different analysis type or check your NumPy version (try downgrading to NumPy 1.26.0).")
        st.info("Available analysis types in limited mode: Exploratory Data Analysis, Supervised Learning (Regression), Time Series Analysis")
    
    # Main content based on analysis type
    elif analysis_type == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        try:
            # Initialize analysis object
            analysis = AnalisisDatosExploratorio(
                uploaded_file,
                delimiter=delimiter_value,
                decimal=decimal_value
            )
            
            # Display general information
            st.subheader("Dataset Information")
            st.dataframe(analysis.data.head())
            
            # Add tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Summary", "Relationships", "Detailed Analysis"])
            
            with tab1:
                st.subheader("Statistical Summary")
                st.dataframe(analysis.data.describe())
                
                # Normalize data
                st.subheader("Normalized Data")
                df_normalized = analysis.normalizar_datos()
                if df_normalized is not None:
                    st.dataframe(df_normalized.head())
            
            with tab2:
                st.subheader("Variable Relationships")
                # This will capture the matplotlib output
                fig, ax = plt.subplots(figsize=(10, 8))
                numeric_data = analysis.data.select_dtypes(include=['int64', 'float64'])
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
                plt.title('Correlation Heatmap')
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Detailed Analysis")
                
                # Use columns for a better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Boxplot
                    st.write("Boxplot of Numerical Variables")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    analysis.data.boxplot(grid=False, vert=False, ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    # Density plot
                    st.write("Density Plot of Numerical Variables")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    analysis.data.plot(kind='density', figsize=(10, 6), linewidth=2, ax=ax)
                    st.pyplot(fig)
                
                # Histogram (full width)
                st.write("Histograms of Numerical Variables")
                fig, ax = plt.subplots(figsize=(10, 6))
                analysis.data.hist(bins=20, edgecolor="black", grid=False, ax=ax)
                st.pyplot(fig)
                
        except Exception as e:
            # Show error in the UI but don't print to terminal
            st.error("An error occurred while processing the data. Please check your file format and settings.")
            # Don't include the traceback or error details to prevent terminal output
    
    elif analysis_type == "Unsupervised Learning":
        st.header("Unsupervised Learning")
        
        if not SKLEARN_EXTRA_AVAILABLE:
            st.warning("scikit-learn-extra is not installed. K-medoids clustering will not be available.")
        
        if not UMAP_AVAILABLE:
            st.warning("umap-learn is not installed. UMAP dimensionality reduction will not be available.")
        
        try:
            # Initialize analysis object
            analysis = AnalisisDatosExploratorio(
                uploaded_file,
                delimiter=delimiter_value,
                decimal=decimal_value,
                tipo_modelo='no_supervisado'
            )
            
            # Display data preview
            st.subheader("Dataset Preview")
            st.dataframe(analysis.data.head())
            
            # Normalize data
            st.subheader("Data Normalization")
            df_normalized = analysis.normalizar_datos()
            
            if df_normalized is not None:
                # Create unsupervised learning object
                unsupervised = NoSupervisados(df=df_normalized)
                
                # Create tabs for different methods
                tab1, tab2, tab3 = st.tabs(["PCA", "Clustering", "Dimensionality Reduction"])
                
                with tab1:
                    st.subheader("Principal Component Analysis (PCA)")
                    
                    n_components = st.slider("Number of components", min_value=2, max_value=10, value=5)
                    
                    if st.button("Run PCA"):
                        # Perform PCA
                        coordinates = unsupervised.ACP(n_componentes=n_components)
                        
                        if coordinates is not None:
                            st.subheader("Principal Components")
                            st.dataframe(coordinates.head())
                            
                            # Plot principal plane
                            st.subheader("Principal Plane")
                            unsupervised.plot_plano_principal(ejes=[0, 1], titulo="Principal Plane")
                            
                            # Plot correlation circle
                            st.subheader("Correlation Circle")
                            unsupervised.plot_circulo(ejes=[0, 1], titulo="Correlation Circle")
                            
                            # Plot biplot
                            st.subheader("Biplot")
                            unsupervised.plot_sobreposicion(ejes=[0, 1], titulo="Biplot: Variables and Individuals")
                
                with tab2:
                    st.subheader("Clustering")
                    
                    # Filter clustering methods based on available packages
                    clustering_methods = ["K-means"]
                    if SKLEARN_EXTRA_AVAILABLE:
                        clustering_methods.append("K-medoids")
                    clustering_methods.append("Hierarchical Agglomerative Clustering (HAC)")
                    
                    clustering_method = st.selectbox(
                        "Select clustering method",
                        clustering_methods
                    )
                    
                    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
                    
                    if st.button("Run Clustering"):
                        if clustering_method == "K-means":
                            clusters = unsupervised.KMEDIAS(n_clusters=n_clusters, metodo='kmeans')
                            st.write("K-means clustering completed.")
                        
                        elif clustering_method == "K-medoids" and SKLEARN_EXTRA_AVAILABLE:
                            clusters = unsupervised.KMEDIAS(n_clusters=n_clusters, metodo='kmedoids')
                            st.write("K-medoids clustering completed.")
                        
                        elif clustering_method == "Hierarchical Agglomerative Clustering (HAC)":
                            linkage_method = st.selectbox(
                                "Select linkage method",
                                ["ward", "average", "complete", "single"]
                            )
                            clusters = unsupervised.HAC(num_clusters=n_clusters, metodo=linkage_method)
                            st.write(f"HAC clustering with {linkage_method} linkage completed.")
                
                with tab3:
                    st.subheader("Dimensionality Reduction")
                    
                    # Filter dimensionality reduction methods based on available packages
                    dim_methods = ["t-SNE"]
                    if UMAP_AVAILABLE:
                        dim_methods.append("UMAP")
                    
                    dim_reduction_method = st.selectbox(
                        "Select dimensionality reduction method",
                        dim_methods
                    )
                    
                    n_components = st.slider("Number of components", min_value=2, max_value=5, value=2)
                    
                    if dim_reduction_method == "UMAP" and UMAP_AVAILABLE:
                        n_neighbors = st.slider("Number of neighbors", min_value=5, max_value=50, value=15)
                    
                    if st.button("Run Dimensionality Reduction"):
                        if dim_reduction_method == "t-SNE":
                            st.write("Running t-SNE...")
                            # Will capture matplotlib output
                            unsupervised.TSNE(n_componentes=n_components)
                        
                        elif dim_reduction_method == "UMAP" and UMAP_AVAILABLE:
                            st.write("Running UMAP...")
                            unsupervised.UMAP(n_componentes=n_components, n_neighbors=n_neighbors)
        
        except Exception as e:
            # Show error in the UI but don't print to terminal
            st.error("An error occurred while processing the data. Please check your file format and settings.")
            # Don't include the traceback or error details to prevent terminal output
    
    elif analysis_type == "Supervised Learning (Regression)":
        st.header("Supervised Learning - Regression")
        
        try:
            # Load data for preview
            df = pd.read_csv(uploaded_file, delimiter=delimiter_value, decimal=decimal_value)
            df = df.dropna()  # Drop null values explicitly
            
            # Display data preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("Dataset needs at least 2 numeric columns for regression analysis")
            else:
                # Select target variable
                target_column = st.selectbox(
                    "Select target variable (Y)",
                    options=numeric_columns,
                    key="target_var"
                )
                
                # Add tabs for different methods
                tab1, tab2 = st.tabs(["Simple Regression", "Model Comparison"])
                
                with tab1:
                    st.subheader("Simple Linear Regression")
                    
                    # Filter out the target from feature options
                    feature_options = [col for col in numeric_columns if col != target_column]
                    
                    # Select feature for simple regression
                    feature_column = st.selectbox(
                        "Select feature variable (X)",
                        options=feature_options,
                        key="feature_var"
                    )
                    
                    if st.button("Run Simple Regression"):
                        # Create a fresh copy of the dataframe with just the columns we need
                        regression_df = df[[target_column, feature_column]].copy()
                        
                        # Split the data
                        X = regression_df[[feature_column]]
                        y = regression_df[target_column]
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=42
                        )
                        
                        # Create and fit the model
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = np.mean(np.abs(y_test - y_pred))
                        relative_error = mae / np.mean(y_test) * 100
                        
                        # Display results
                        st.subheader("Model Coefficients")
                        st.write(f"Coefficient: {model.coef_[0]:.4f}")
                        st.write(f"Intercept: {model.intercept_:.4f}")
                        st.write(f"Formula: {target_column} = {model.coef_[0]:.4f} × {feature_column} + {model.intercept_:.4f}")
                        
                        # Display metrics
                        st.subheader("Regression Metrics")
                        metrics_df = pd.DataFrame({
                            "R²": [r2],
                            "RMSE": [rmse],
                            "MAE": [mae],
                            "Relative Error (%)": [relative_error]
                        })
                        st.dataframe(metrics_df)
                        
                        # Visualize relationship
                        st.subheader(f"Relationship: {feature_column} vs {target_column}")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Scatter plot of actual data
                        ax.scatter(X, y, color='blue', alpha=0.5, label='Data points')
                        
                        # Sort for line plot
                        sorted_indices = np.argsort(X.values.flatten())
                        X_sorted = X.values[sorted_indices]
                        
                        # Regression line
                        ax.plot(
                            X_sorted, 
                            model.intercept_ + model.coef_[0] * X_sorted,
                            color='red', linewidth=2, label='Regression line'
                        )
                        
                        ax.set_xlabel(feature_column)
                        ax.set_ylabel(target_column)
                        ax.set_title(f'Linear Regression: {feature_column} vs {target_column}')
                        ax.legend()
                        ax.grid(alpha=0.3)
                        
                        st.pyplot(fig)
                
                with tab2:
                    st.subheader("Model Comparison")
                    
                    # Select features for model comparison
                    feature_selection = st.multiselect(
                        "Select features for model comparison",
                        options=feature_options,
                        default=feature_options[:min(3, len(feature_options))]
                    )
                    
                    if len(feature_selection) == 0:
                        st.warning("Please select at least one feature for model comparison")
                    elif st.button("Compare Regression Models"):
                        with st.spinner("Training and comparing models..."):
                            # Create a fresh copy with selected columns
                            model_df = df[[target_column] + feature_selection].copy()
                            
                            # Split the data
                            X = model_df[feature_selection]
                            y = model_df[target_column]
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.25, random_state=42
                            )
                            
                            # Models to compare
                            models = {
                                "Linear Regression": LinearRegression(),
                                "Ridge": Ridge(alpha=1.0),
                                "Lasso": Lasso(alpha=0.1),
                                "Decision Tree": DecisionTreeRegressor(max_depth=5),
                                "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
                                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100)
                            }
                            
                            # Train and evaluate models
                            results = []
                            
                            for name, model in models.items():
                                # Train model
                                model.fit(X_train, y_train)
                                
                                # Predict
                                y_pred = model.predict(X_test)
                                
                                # Calculate metrics
                                r2 = r2_score(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                mae = np.mean(np.abs(y_test - y_pred))
                                relative_error = mae / np.mean(y_test) * 100
                                
                                # Add to results
                                results.append({
                                    "Model": name,
                                    "R²": r2,
                                    "RMSE": rmse,
                                    "MAE": mae,
                                    "Relative Error (%)": relative_error
                                })
                            
                            # Create DataFrame with results
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Model Comparison Results")
                            st.dataframe(results_df)
                            
                            # Plot comparison
                            st.subheader("Visual Comparison")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Set up bar positions
                            models_count = len(results_df)
                            metric_columns = ["R²", "RMSE", "MAE", "Relative Error (%)"]
                            x = np.arange(len(metric_columns))
                            width = 0.8 / models_count
                            
                            # Plot bars for each model
                            for i, (_, row) in enumerate(results_df.iterrows()):
                                ax.bar(
                                    x + (i - models_count/2 + 0.5) * width,
                                    [row[metric] for metric in metric_columns],
                                    width,
                                    label=row["Model"]
                                )
                            
                            # Customize plot
                            ax.set_ylabel('Value')
                            ax.set_title('Model Comparison by Metrics')
                            ax.set_xticks(x)
                            ax.set_xticklabels(metric_columns)
                            ax.legend(loc='best')
                            ax.grid(alpha=0.3)
                            
                            st.pyplot(fig)
        
        except Exception as e:
            # Show error in the UI but don't print to terminal
            st.error("An error occurred while processing the data. Please check your file format and settings.")
            # Don't include the traceback or error details to prevent terminal output
    
    elif analysis_type == "Supervised Learning (Classification)":
        st.header("Supervised Learning - Classification")
        
        try:
            # Load data for preview
            df = pd.read_csv(uploaded_file, delimiter=delimiter_value, decimal=decimal_value)
            df = df.dropna()  # Drop null values explicitly
            
            # Display data preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Get columns that could be used as classification targets
            # We'll look for columns with a small number of unique values (potential class labels)
            potential_targets = []
            for col in df.columns:
                n_unique = df[col].nunique()
                # Consider columns with 2-10 unique values as potential classification targets
                if 2 <= n_unique <= 10:
                    potential_targets.append(col)
            
            if not potential_targets:
                st.warning("No suitable classification target columns found. A good classification target should have 2-10 unique values.")
                # Add all columns as a fallback
                potential_targets = df.columns.tolist()
            
            # Select target variable
            target_column = st.selectbox(
                "Select target variable (class labels)",
                options=potential_targets,
                key="classification_target"
            )
            
            # Show unique values in the target
            unique_values = df[target_column].unique()
            st.write(f"Unique values in target column: {', '.join(map(str, unique_values))}")
            
            # Add tabs for EDA and model comparison
            tab1, tab2 = st.tabs(["Exploratory Analysis", "Model Comparison"])
            
            with tab1:
                st.subheader("Exploratory Data Analysis")
                
                if st.button("Run EDA"):
                    # Basic statistics
                    st.write("Shape of the data:", df.shape)
                    
                    # Show target distribution
                    st.subheader(f"Distribution of Target Variable: {target_column}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[target_column].value_counts().plot(kind='bar', ax=ax)
                    plt.title(f"Class Distribution: {target_column}")
                    plt.xlabel("Class")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistical summary
                    st.subheader("Statistical Summary")
                    st.dataframe(df.describe().round(2))
                    
                    # Correlation heatmap
                    st.subheader("Correlation Matrix")
                    numeric_df = df.select_dtypes(include=['number'])
                    if numeric_df.shape[1] > 1:  # Ensure we have at least 2 numeric columns
                        fig, ax = plt.subplots(figsize=(12, 10))
                        correlation = numeric_df.corr()
                        sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Not enough numeric columns for correlation analysis.")
            
            with tab2:
                st.subheader("Model Comparison")
                
                # Get features (exclude the target)
                features = [col for col in df.columns if col != target_column]
                
                # Allow selection of features
                selected_features = st.multiselect(
                    "Select features for model training",
                    options=features,
                    default=features[:min(5, len(features))]  # Default to first 5 features or fewer
                )
                
                # Select test size
                test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
                
                if len(selected_features) == 0:
                    st.warning("Please select at least one feature for model training")
                elif st.button("Compare Classification Models"):
                    with st.spinner("Training and comparing models... This may take a while."):
                        # Create a fresh copy with selected columns
                        model_df = df[[target_column] + selected_features].copy()
                        
                        # Check if target variable is numeric or categorical
                        if pd.api.types.is_numeric_dtype(model_df[target_column]):
                            # For numeric target columns, we'll treat as binary classification if there are only 2 values
                            unique_values = model_df[target_column].unique()
                            if len(unique_values) == 2:
                                st.write(f"Binary classification with values: {unique_values}")
                            elif len(unique_values) > 10:
                                st.warning(f"The target column has {len(unique_values)} unique values, which may be too many for classification. Consider using regression instead.")
                        
                        # Split the data
                        X = model_df[selected_features]
                        y = model_df[target_column]
                        
                        # Apply preprocessing
                        from sklearn.preprocessing import StandardScaler, LabelEncoder
                        
                        # Handle categorical features
                        from sklearn.compose import ColumnTransformer
                        from sklearn.preprocessing import OneHotEncoder
                        
                        categorical_features = X.select_dtypes(include=['object', 'category']).columns
                        numeric_features = X.select_dtypes(include=['number']).columns
                        
                        # Create preprocessing pipeline
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_features),
                                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                            ]
                        )
                        
                        # Encode the target if it's categorical
                        if not pd.api.types.is_numeric_dtype(y):
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            # Display the encoding mapping
                            st.subheader("Target Variable Encoding")
                            for i, class_name in enumerate(le.classes_):
                                st.write(f"{class_name} → {i}")
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Apply preprocessing
                        X_train_processed = preprocessor.fit_transform(X_train)
                        X_test_processed = preprocessor.transform(X_test)
                        
                        # Import classification models
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.svm import SVC
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.naive_bayes import GaussianNB
                        
                        # Import metrics
                        from sklearn.metrics import (
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                            classification_report, roc_auc_score
                        )
                        
                        # Models to compare
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                            "SVM": SVC(probability=True, random_state=42),
                            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                            "Naive Bayes": GaussianNB()
                        }
                        
                        # Train and evaluate models
                        results = []
                        confusion_matrices = {}
                        roc_data = {}
                        
                        for name, model in models.items():
                            # Train model
                            model.fit(X_train_processed, y_train)
                            
                            # Predict
                            y_pred = model.predict(X_test_processed)
                            y_prob = model.predict_proba(X_test_processed)[:, 1] if len(np.unique(y)) == 2 else None
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Binary classification metrics
                            if len(np.unique(y)) == 2:
                                precision = precision_score(y_test, y_pred)
                                recall = recall_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred)
                                try:
                                    auc_score = roc_auc_score(y_test, y_prob)
                                except:
                                    auc_score = np.nan
                                    
                                # Store ROC curve data
                                try:
                                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                                    roc_data[name] = (fpr, tpr, auc_score)
                                except:
                                    pass
                            else:
                                # Multiclass metrics
                                precision = precision_score(y_test, y_pred, average='weighted')
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')
                                auc_score = np.nan
                            
                            # Add to results
                            results.append({
                                "Model": name,
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1 Score": f1,
                                "AUC": auc_score if len(np.unique(y)) == 2 else np.nan
                            })
                            
                            # Store confusion matrix
                            cm = confusion_matrix(y_test, y_pred)
                            confusion_matrices[name] = cm
                        
                        # Create DataFrame with results
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Model Comparison Results")
                        st.dataframe(results_df)
                        
                        # Visualize results
                        if len(np.unique(y)) == 2:  # Binary classification
                            # ROC Curves
                            st.subheader("ROC Curves")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            for name, (fpr, tpr, auc_score) in roc_data.items():
                                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                            
                            ax.plot([0, 1], [0, 1], 'k--', label='Random')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curves for Different Models')
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Display confusion matrices
                        st.subheader("Confusion Matrices")
                        
                        # Create grid layout for confusion matrices
                        n_models = len(confusion_matrices)
                        cols_per_row = 2
                        n_rows = (n_models + cols_per_row - 1) // cols_per_row
                        
                        for i in range(0, n_models, cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, name in enumerate(list(confusion_matrices.keys())[i:i+cols_per_row]):
                                if i+j < n_models:
                                    with cols[j]:
                                        st.write(f"**{name}**")
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        disp = ConfusionMatrixDisplay(
                                            confusion_matrix=confusion_matrices[name],
                                            display_labels=np.unique(y)
                                        )
                                        disp.plot(ax=ax, cmap='Blues', values_format='d')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                        
                        # Feature importance for applicable models
                        st.subheader("Feature Importance")
                        feature_importances = {}
                        
                        # Get feature names after preprocessing
                        feature_names = list(numeric_features)
                        if len(categorical_features) > 0:
                            # Get encoded feature names
                            encoder = preprocessor.named_transformers_['cat']
                            if hasattr(encoder, 'get_feature_names_out'):
                                encoded_features = encoder.get_feature_names_out(categorical_features)
                                feature_names.extend(encoded_features)
                        
                        # Get feature importance for models that support it
                        for name, model in models.items():
                            if hasattr(model, 'feature_importances_'):
                                # For tree-based models
                                feature_importances[name] = model.feature_importances_
                            elif name == "Logistic Regression":
                                # For logistic regression
                                if hasattr(model, 'coef_'):
                                    feature_importances[name] = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                        
                        # Visualize feature importance
                        if feature_importances:
                            for name, importance in feature_importances.items():
                                if len(importance) == len(feature_names):
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sorted_idx = np.argsort(importance)
                                    
                                    # Show top 15 features or all if less than 15
                                    n_features = min(15, len(feature_names))
                                    plt.barh(range(n_features), importance[sorted_idx[-n_features:]])
                                    plt.yticks(range(n_features), [feature_names[i] for i in sorted_idx[-n_features:]])
                                    plt.xlabel('Feature Importance')
                                    plt.title(f'Feature Importance - {name}')
                                    st.pyplot(fig)
                                else:
                                    st.write(f"Feature importance for {name} cannot be displayed (dimension mismatch)")
                        
                        st.success("Model comparison completed!")
        
        except Exception as e:
            # Show error in the UI but don't print to terminal
            st.error("An error occurred while processing the data. Please check your file format and settings.")
            # Don't include the traceback or error details to prevent terminal output
    
    elif analysis_type == "Time Series Analysis":
        st.header("Time Series Analysis")
        
        try:
            # Get a preview of the data for column selection
            df_preview = pd.read_csv(uploaded_file, delimiter=delimiter_value, decimal=decimal_value)
            
            # Display data preview
            st.subheader("Dataset Preview")
            st.dataframe(df_preview.head())
            
            # Column selection
            date_column = st.selectbox(
                "Select date column",
                df_preview.columns
            )
            
            value_column = st.selectbox(
                "Select value column",
                df_preview.select_dtypes(include=['int64', 'float64']).columns
            )
            
            # Initialize time series object with correct delimiter type
            delimiter_type = 1 if delimiter_value == "," else 2
            
            # Create a temporary CSV file with the processed data to avoid delimiter issues
            with st.spinner("Processing data..."):
                # Process the dataframe directly instead of having SerieTiempo read the file again
                # Convert date column to datetime
                df_preview[date_column] = pd.to_datetime(df_preview[date_column], errors='coerce')
                
                # Drop rows with invalid dates
                df_preview = df_preview.dropna(subset=[date_column])
                
                # Create a direct time series from the dataframe
                ts_analysis = SerieTiempo(
                    columna_fecha=date_column,
                    columna_valor=value_column,
                    delimiter_type=delimiter_type
                )
                
                # Create a method to work with our dataframe directly
                def create_series_from_df(df):
                    fechas = pd.DatetimeIndex(df[date_column])
                    ts = pd.Series(df[value_column].values, index=fechas)
                    return ts
            
            # Add tabs for different operations
            tab1, tab2, tab3 = st.tabs(["Data Preparation", "Time Series Creation", "Forecasting"])
            
            with tab1:
                st.subheader("Data Preparation")
                
                # Show data types
                st.write("Data Types:")
                for col, dtype in df_preview.dtypes.items():
                    st.write(f"{col}: {dtype}")
                
                # Show null values
                st.write("Null Values:")
                st.write(df_preview.isna().sum())
                
                # Option to remove invalid dates
                if st.button("Remove Invalid Dates"):
                    df_preview = df_preview.dropna(subset=[date_column])
                    st.success("Invalid dates removed successfully")
                    st.dataframe(df_preview.head())
            
            with tab2:
                st.subheader("Time Series Creation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Start date
                    min_date = df_preview[date_column].min()
                    if pd.notna(min_date):
                        start_date = st.date_input("Start Date", min_date)
                    else:
                        start_date = st.date_input("Start Date")
                
                with col2:
                    # End date
                    max_date = df_preview[date_column].max()
                    if pd.notna(max_date):
                        end_date = st.date_input("End Date", max_date)
                    else:
                        end_date = st.date_input("End Date")
                
                # Frequency selection
                frequency = st.selectbox(
                    "Select frequency",
                    ["D", "W", "M", "Q", "H", "10min", "30min", "1h"]
                )
                
                if st.button("Generate Date Range"):
                    # Filter the dataframe to the selected date range
                    mask = (df_preview[date_column] >= pd.Timestamp(start_date)) & (df_preview[date_column] <= pd.Timestamp(end_date))
                    filtered_df = df_preview[mask].copy()
                    
                    # Check if we have data in the range
                    if filtered_df.empty:
                        st.warning(f"No data found between {start_date} and {end_date}")
                    else:
                        st.success(f"Found {len(filtered_df)} data points in the selected date range")
                        st.dataframe(filtered_df.head())
                        
                        # Check for missing dates
                        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
                        existing_dates = pd.DatetimeIndex(filtered_df[date_column].values)
                        missing_dates = [x for x in date_range if x not in existing_dates]
                        
                        if missing_dates:
                            st.write(f"Found {len(missing_dates)} missing dates in the selected range")
                            missing_df = pd.DataFrame({date_column: missing_dates})
                            st.dataframe(missing_df.head())
                            
                            # Option to impute missing dates
                            if st.button("Impute Missing Dates"):
                                # Create a complete time series with all dates
                                complete_dates = pd.DataFrame({date_column: date_range})
                                
                                # Merge with existing data
                                merged_df = pd.merge(complete_dates, filtered_df, on=date_column, how='left')
                                
                                # Fill missing values using interpolation
                                merged_df[value_column] = merged_df[value_column].interpolate(method='linear')
                                
                                st.success("Missing dates imputed successfully")
                                st.dataframe(merged_df.head())
                                
                                # Update the filtered dataframe
                                filtered_df = merged_df
                        else:
                            st.success("No missing dates found in the selected range")
                
                # Create time series
                if st.button("Create Time Series"):
                    try:
                        # Create a time series from the filtered dataframe
                        if 'filtered_df' in locals():
                            ts = create_series_from_df(filtered_df)
                        else:
                            ts = create_series_from_df(df_preview)
                        
                        if ts is not None and not ts.empty:
                            st.success("Time series created successfully")
                            
                            # Plot the time series
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ts.plot(ax=ax)
                            plt.title('Time Series')
                            plt.ylabel(value_column)
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Statistics
                            st.subheader("Time Series Statistics")
                            stats = pd.DataFrame({
                                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                'Value': [
                                    ts.mean(),
                                    ts.median(),
                                    ts.std(),
                                    ts.min(),
                                    ts.max()
                                ]
                            })
                            st.dataframe(stats)
                        else:
                            st.error("Could not create time series - no valid data found")
                    except Exception as e:
                        st.error("An error occurred while creating the time series. Please check your file format and settings.")
            
            with tab3:
                st.subheader("Time Series Forecasting")
                
                st.info("Note: Advanced forecasting models (ARIMA, LSTM, etc.) require additional libraries that are not included in this version. Basic forecasting capabilities are available.")
                
                # Number of periods to predict
                forecast_periods = st.slider(
                    "Number of periods to forecast",
                    min_value=1,
                    max_value=30,
                    value=7
                )
                
                forecast_methods = ["Simple Moving Average", "Exponential Smoothing"]
                forecast_method = st.selectbox("Select forecasting method", forecast_methods)
                
                if st.button("Run Forecast"):
                    try:
                        # Create a time series from the dataframe
                        if 'filtered_df' in locals():
                            ts = create_series_from_df(filtered_df)
                        else:
                            ts = create_series_from_df(df_preview)
                        
                        if ts is not None and not ts.empty:
                            # Train-test split
                            train_size = int(len(ts) * 0.8)
                            train = ts[:train_size]
                            test = ts[train_size:]
                            
                            # Forecast based on selected method
                            if forecast_method == "Simple Moving Average":
                                # Calculate moving average
                                window = min(7, len(train))
                                ma = train.rolling(window=window).mean()
                                
                                # Forecast using the last moving average value
                                last_ma = ma.iloc[-1]
                                forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                                forecast = pd.Series([last_ma] * forecast_periods, index=forecast_index)
                                
                            elif forecast_method == "Exponential Smoothing":
                                # Simple exponential smoothing
                                alpha = 0.3  # Smoothing parameter
                                smooth = train.copy()
                                for i in range(1, len(train)):
                                    smooth.iloc[i] = alpha * train.iloc[i] + (1 - alpha) * smooth.iloc[i-1]
                                
                                # Forecast using the last smoothed value
                                last_value = smooth.iloc[-1]
                                forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                                forecast = pd.Series([last_value] * forecast_periods, index=forecast_index)
                            
                            # Plot the results
                            fig, ax = plt.subplots(figsize=(12, 6))
                            train.plot(label='Training Data', ax=ax)
                            test.plot(label='Test Data', ax=ax)
                            forecast.plot(label='Forecast', ax=ax, style='--')
                            plt.title(f'Time Series Forecast ({forecast_method})')
                            plt.ylabel(value_column)
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Evaluation metrics
                            if len(test) > 0:
                                st.subheader("Forecast Evaluation")
                                # Calculate error metrics for the test period
                                # Use the last training value as a naive forecast for the test period
                                naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
                                
                                mae = np.mean(np.abs(test - naive_forecast))
                                mse = np.mean((test - naive_forecast) ** 2)
                                rmse = np.sqrt(mse)
                                
                                metrics = pd.DataFrame({
                                    'Metric': ['MAE', 'MSE', 'RMSE'],
                                    'Value': [mae, mse, rmse]
                                })
                                st.dataframe(metrics)
                        else:
                            st.error("Could not create time series - no valid data found")
                    except Exception as e:
                        st.error("An error occurred while forecasting. Please check your file format and settings.")
        
        except Exception as e:
            # Show error in the UI but don't print to terminal
            st.error("An error occurred while processing the data. Please check your file format and settings.")
            # Don't include the traceback or error details to prevent terminal output

else:
    # Instructions when no file is uploaded
    st.info("Please upload a CSV file to begin analysis.")
    
    # Description of the functionality
    st.markdown("""
    ## Data Analysis Framework
    
    This application integrates with the framework_Final.py module to perform various data analysis tasks:
    
    - **Exploratory Data Analysis**: Statistical summaries, data visualization, correlation analysis
    - **Unsupervised Learning**: PCA, clustering (K-means, K-medoids, HAC), dimensionality reduction
    - **Supervised Learning (Regression)**: Linear regression, model comparison
    - **Supervised Learning (Classification)**: Classification model evaluation and comparison
    - **Time Series Analysis**: Data preparation, time series creation, forecasting
    
    ### How to use
    
    1. Select an analysis type from the sidebar
    2. Upload a CSV file
    3. Select the appropriate delimiter and decimal separator
    4. Configure the analysis options
    5. View the results
    """) 