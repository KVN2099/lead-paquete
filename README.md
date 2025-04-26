# Data Analysis Streamlit App

A comprehensive data analysis application built with Streamlit that leverages the `framework_Final.py` module for various data analysis tasks.

## Features

- **Exploratory Data Analysis**: Statistical summaries, data visualization, correlation analysis
- **Unsupervised Learning**: PCA, clustering (K-means, K-medoids, HAC), dimensionality reduction (t-SNE, UMAP)
- **Supervised Learning (Regression)**: Simple regression, model comparison with various algorithms
- **Supervised Learning (Classification)**: Classification model evaluation and comparison
- **Time Series Analysis**: Data preparation, time series creation, forecasting

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Access the app in your web browser (the URL will be displayed in the terminal)
3. Select an analysis type from the sidebar
4. Upload your CSV file
5. Select the appropriate delimiter and decimal separator
6. Configure the analysis options and run the desired analysis

## Required Data Formats

- **Exploratory Data Analysis**: Any CSV file with numerical columns
- **Unsupervised Learning**: CSV file with numerical features
- **Supervised Learning (Regression)**: CSV file with numerical target variable and features
- **Supervised Learning (Classification)**: CSV file with a binary target column (preferably named 'Bankrupt?')
- **Time Series Analysis**: CSV file with date column and value column

## Dependencies

The application requires the following main libraries:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- prince
- sklearn_extra
- scipy
- umap

These dependencies are included in the `requirements.txt` file.

## Note

Make sure that `framework_Final.py` is in the same directory as the app.py file.