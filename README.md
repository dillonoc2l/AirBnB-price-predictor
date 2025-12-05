This project builds a machine learning model to predict Airbnb listing prices across London using a Gradient Boosting Regressor trained on cleaned Airbnb listing data. The workflow includes preprocessing steps such
as one-hot encoding categorical features, engineering a geographic distance-to-city-centre feature using the Haversine formula, log-transforming prices to reduce outlier impact, and performing hyperparameter optimisation
with RandomizedSearchCV to select the best model. A Streamlit web application provides an interactive interface where users can enter listing attributes—such as location, neighbourhood, room type, and host metrics—and
receive an estimated nightly price. The system also includes automatic model evaluation and saving logic to ensure the best-performing model is stored for deployment. This project showcases end-to-end model development,
feature engineering, optimisation, and UI deployment for a real-world pricing prediction task.
