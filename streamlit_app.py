import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv("data/train.csv")

# Define the feature columns
feature_cols = ['Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
                'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index',
                'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'X_Distance',
                'Y_Distance', 'Relative_Perimeter', 'Circularity', 'Color_Contrast',
                'Combined_Index', 'Sigmoid_Areas']

# Define the min, max, mean, and std values for each feature column
min_values = [0.0, 39.0, 1227.0, 3.688880, 0.0, 0.0, 0.008300, 0.014400, 0.105000, 0.0, -1128.0, -4.609101, 0.316359, 0.104471, -54.0, -0.757648, 0.685290]
max_values = [196.0, 253.0, 1794.0, 5.703783, 0.995200, 0.927500, 1.000000, 1.000000, 1.000000, 1.000000, 1665.0, 4.609052, 0.999990, 7.571050, 223.0, 0.554388, 0.989588]
mean_values = [84.808419, 128.647380, 1459.350747, 4.170039, 0.352939, 0.409309, 0.574520, 0.614749, 0.831652, 0.591899, 44.002966, -0.000160, 0.511639, 0.508162, 43.838961, -0.020341, 0.906873]
std_values = [28.800344, 14.196976, 145.568687, 0.523943, 0.318976, 0.124143, 0.259436, 0.222391, 0.220966, 0.482050, 80.266662, 0.314324, 0.038082, 0.209828, 26.573444, 0.089863, 0.049516]

# Create the Streamlit app
st.title('Prediction using the Trained Model')


# Create input sliders for the features
st.text("The slider min and max values are matched according to the training dataset.")
input_data = []
for i, col in enumerate(feature_cols):
    input_val = st.slider(col, min_values[i], max_values[i], mean_values[i], )
    input_data.append(input_val)

# Make the prediction
if st.button('Predict'):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    st.write(f'The prediction is: {prediction[0]}')