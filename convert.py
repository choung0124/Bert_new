import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have created a list of dictionaries called 'data' with the desired format
data = [
    {
        "sentence": "John bought a car for Mary.",
        "subject_entity": "John",
        "object_entity": "car",
        "relationship": "bought"
    },
    # More data points...
]

# Convert the list of dictionaries to a DataFrame
data_df = pd.DataFrame(data)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data_df, test_size=0.2, random_state=42)
