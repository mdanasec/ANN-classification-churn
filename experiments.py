# Pandas is used to read, manipulate, and work with tabular data such as CSV files
import pandas as pd
from PIL.ImageOps import fit


# train_test_split is used to divide the dataset into training and testing data
from sklearn.model_selection import train_test_split

# StandardScaler -> scales numerical features to a similar range
# LabelEncoder -> converts categorical values into numerical values
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Pickle is used to save Python objects such as encoders and scalers
# so that we can reuse them later
import pickle


# -------------------- READ DATASET --------------------

# Read the CSV file and store it in a Pandas DataFrame
data = pd.read_csv('Churn_Modelling.csv')

# Display the first 5 rows to understand the structure of the dataset
print(data.head())


# -------------------- REMOVE UNNECESSARY COLUMNS --------------------

# Remove columns that are not useful for predicting customer churn
# RowNumber -> only represents the row number
# CustomerId -> unique ID of the customer
# Surname -> customer's surname
# axis=1 means we are removing columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# -------------------- LABEL ENCODING --------------------

# Create a LabelEncoder object
# It converts categorical values into numerical values
label_encoder_gender = LabelEncoder()

# Convert Gender values into numbers
# Example:
# Female -> 0
# Male   -> 1
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# Display the dataset after converting Gender into numbers
print(data)


# -------------------- ONE-HOT ENCODING --------------------

# Import OneHotEncoder
# It converts categorical values into separate binary columns
from sklearn.preprocessing import OneHotEncoder

# Create a OneHotEncoder object
onehot_encoder_geo = OneHotEncoder()

# Learn the different Geography categories and convert them into encoded values
#
# Example:
# France  -> [1, 0, 0]
# Germany -> [0, 1, 0]
# Spain   -> [0, 0, 1]
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])

# Convert the encoded sparse matrix into a normal NumPy array
# and display it
print(geo_encoder.toarray())


# Create a DataFrame from the encoded Geography values
# get_feature_names_out() creates column names such as:
# Geography_France, Geography_Germany, Geography_Spain
geo_encoded_def = pd.DataFrame(
    geo_encoder.toarray(),
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Display the One-Hot encoded Geography DataFrame
print(geo_encoded_def)


# -------------------- COMBINE ENCODED DATA --------------------

# Remove the original Geography column
# and combine the new One-Hot encoded Geography columns with the dataset
data = pd.concat(
    [data.drop('Geography', axis=1), geo_encoded_def],
    axis=1
)

# Display the final dataset after encoding
print(data.head())


# -------------------- SAVE ENCODERS --------------------

# Save the Gender LabelEncoder into a file
# We will reuse this encoder later when making predictions on new data
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)


# Save the Geography OneHotEncoder into a file
# We will reuse the same encoder later for new/prediction data
with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)


# -------------------- DIVIDE FEATURES AND TARGET --------------------

# X = independent/input features
# These are the features that the ANN will use to make a prediction
x = data.drop('Exited', axis=1)

# y = dependent/target feature
# This is the value that the ANN needs to predict
#
# Exited:
# 0 -> Customer did not leave the bank
# 1 -> Customer left the bank
y = data['Exited']


# -------------------- TRAIN-TEST SPLIT --------------------

# Divide the data into training and testing sets
#
# X_train -> input data used for training
# X_test  -> input data used for testing
# y_train -> correct answers used during training
# y_test  -> correct answers used to evaluate the model
#
# test_size=0.2 means:
# 80% data -> training
# 20% data -> testing
#
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------- FEATURE SCALING --------------------

# Create a StandardScaler object
# It brings numerical features to a similar scale
scaler = StandardScaler()

# Learn the scaling information from training data
# and then transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the testing data using the SAME scaler
# We do not use fit_transform() here because the scaler
# must use the information learned from the training data
X_test = scaler.transform(X_test)

# Display the scaled training data
print(X_train)


# -------------------- SAVE SCALER --------------------

# Save the trained scaler into a file
# We can reuse the same scaler later when making predictions
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# ----------------Artificial Neural Network IMPLEMENTATION------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

(X_train.shape[1],)

model = Sequential([
    Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)), #HL1 connected with input layer
    Dense(32,activation='relu'), #HL2
    Dense(1, activation='sigmoid')# output layer
])
print(model.summary())

import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

# compile the model
model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# Set up the Tensorboard
log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Set up Early Stopping
early_stopping_callback= EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ------------------------
# Train the model
#-------------------------

history = model.fit(
    X_train, y_train, validation_data =(X_test, y_test), epochs =100,
    callbacks = [tensorflow_callback, early_stopping_callback]
)

model.save('model.h5')







