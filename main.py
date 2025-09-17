Creating a data bias detector can be quite an involved task, so I'll provide a simplified version that identifies potential biases in a given dataset and visualizes them. This program will focus on identifying imbalances in class distribution and feature distributions, which are common indicators of bias. We'll use a combination of Python libraries such as `pandas`, `matplotlib`, and `seaborn` to perform our analyses and visualizations.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}.")
        return data
    except FileNotFoundError as e:
        print(f"Error: The file {file_path} was not found.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

# Function to visualize class distribution
def plot_class_distribution(data, target_column):
    try:
        sns.countplot(data[target_column])
        plt.title('Class Distribution')
        plt.show()
    except KeyError as e:
        print(f"Error: The target column '{target_column}' does not exist in the dataset.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while plotting class distribution: {e}")
        raise e

# Function to visualize feature distribution by class
def plot_feature_distribution(data, feature_column, target_column):
    try:
        sns.histplot(data, x=feature_column, hue=target_column, multiple='stack')
        plt.title(f'Distribution of {feature_column} by {target_column}')
        plt.show()
    except KeyError as e:
        print(f"Error: One of the columns '{feature_column}' or '{target_column}' does not exist in the dataset.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while plotting feature distribution: {e}")
        raise e

# Function to calculate and print basic statistics
def calculate_statistics(data, target_column):
    try:
        class_counts = data[target_column].value_counts(normalize=True) * 100
        print(f'Class Distribution (%):\n{class_counts}\n')
        
        # Additional stats for continuous features
        print("Feature Statistics:\n")
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                print(f"{column}:\n{data[column].describe()}\n")
    except KeyError as e:
        print(f"Error: The target column '{target_column}' does not exist in the dataset.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while calculating statistics: {e}")
        raise e

# Main function to execute the flow
def main(file_path, target_column):
    data = load_dataset(file_path)
    calculate_statistics(data, target_column)
    plot_class_distribution(data, target_column)
    
    # Plot distribution for each feature
    for column in data.columns:
        if column != target_column and data[column].dtype in [np.float64, np.int64, object]:
            plot_feature_distribution(data, column, target_column)

# Example usage:
# Provide the path to your dataset file and specify the target column
# main('your_dataset.csv', 'target_column_name')

# Uncomment the following line for actual usage, ensure 'file_path' and 'target_column' are correctly set.
# main('your_dataset.csv', 'target_column_name')
```

### Key Points:
- **Data Loading and Error Handling**: The program attempts to load a dataset from a provided CSV file path. It handles file not found errors and other unforeseen exceptions during loading.
- **Class Distribution Plot**: Visualizes class distribution to identify class imbalances, which are common indicators of bias.
- **Feature Distribution Plot**: Visualizes the distribution of features grouped by class labels.
- **Statistical Summary**: Provides a statistical summary of each feature, including descriptive statistics for continuous features.
- **Error Handling**: Ensures robust error checking and reporting throughout the program, allowing better troubleshooting.

### Instructions for Use:
- Replace `'your_dataset.csv'` and `'target_column_name'` with the actual path to your dataset and the name of the target column you want to analyze.
- Ensure you have the required libraries installed: pandas, matplotlib, and seaborn. You can install them via pip:
  ```
  pip install pandas matplotlib seaborn
  ```

This program gives a basic structure for detecting dataset biases, which can be extended with more sophisticated checks and metrics as needed.