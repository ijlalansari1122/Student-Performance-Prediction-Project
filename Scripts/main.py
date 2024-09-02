from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_model

# File path to the CSV file
data_path = r'C:\Users\poono\Desktop\New folder (4)\Student-Performance-Prediction-Project\data\student-scores.csv'

# Call the preprocessing function
df_processed = preprocess_data(data_path)

# Define the target column (choose based on your needs)
target_column = 'combined_score'  # Or use a specific score like 'math_score'

# Call the model training and evaluation function
train_and_evaluate_model(df_processed, target_column)
