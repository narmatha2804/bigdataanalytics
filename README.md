# Item Outlet Sales Prediction

## Description
This project demonstrates how to use a Linear Regression model to predict sales (`Item_Outlet_Sales`) based on various features.

## Dataset
`train.csv`: Contains features and the target variable (`Item_Outlet_Sales`).
`test.csv`: Contains features but not the target variable. The task is to predict `Item_Outlet_Sales`.

## Dependencies
- pandas
- matplotlib
- seaborn
- scikit-learn

Install these dependencies using:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/narmatha2804/bigadataanalytics.git
    ```

2. Run Script:
    ```bash
    python main.py
    ```

## Features
- Predicts sales (`Item_Outlet_Sales`) using a Linear Regression model.
- Includes data preprocessing and model training steps.
- Generates the following visualizations:
  - Feature Importance: Bar chart showing the contribution of each feature.
  - Actual vs. Predicted Sales: Scatter plot to compare predictions with actual sales values.
  - Residual Analysis: Histogram to visualize errors between predicted and actual values.
- Computes performance metrics like RMSE (Root Mean Squared Error) for both training and testing datasets.

## Results
1. Performance Metrics:
   - Training RMSE: Error metric for the training dataset.
   - Testing RMSE: Error metric for the testing dataset.
2. Visualizations:
   - Feature Importance (Bar Chart)
   - Actual vs. Predicted Sales (Scatter Plot)
   - Residual Distribution (Histogram)

## License
This project is licensed under the MIT License.

## Acknowledgements
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/)
- [seaborn Documentation](https://seaborn.pydata.org/)
