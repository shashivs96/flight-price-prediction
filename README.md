#  Flight Price Prediction

This project focuses on predicting flight ticket prices using machine learning techniques. It includes exploratory data analysis (EDA), feature engineering, and model training to develop a regression model that accurately estimates flight prices based on features like airline, source, destination, duration, and more.

##  Project Structure

- `flight_price.ipynb`: Jupyter notebook containing all steps — data preprocessing, EDA, model training, and evaluation.

##  Dataset

- **Source**: Flight fare prediction dataset (typically available on Kaggle or similar platforms)
- **Features Include**:
  - Airline
  - Source and Destination
  - Date of Journey
  - Duration
  - Total Stops
  - Additional Info
- **Target**: Flight ticket price (`Price`)

##  Tools & Libraries Used

- Python (Google Colab / Jupyter)
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `scikit-learn`

##  Model

- **Problem Type**: Regression
- **Model Used**: (e.g., Random Forest Regressor or XGBoost — update based on your notebook)
- **Preprocessing**:
  - Categorical encoding (One-Hot / Label Encoding)
  - Feature extraction from datetime fields
  - Handling missing values
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

##  Performance

- The model was trained and tested with a train-test split.
- Performance is evaluated using regression metrics to assess prediction accuracy.

##  Future Enhancements

- Try advanced models like XGBoost, LightGBM
- Hyperparameter tuning for better performance
- Use pipeline and GridSearchCV
- Deploy the model using Flask or Streamlit

##  Disclaimer

This project is for educational purposes. The dataset used is publicly available and anonymized.

