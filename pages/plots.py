import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

train_df = pd.read_csv('data/train.csv')
st.text("Overview of the training data")
st.dataframe(train_df)

# Assuming train_df is already defined
train_df['Fault_Type'] = train_df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']].idxmax(axis=1)

# Visualization to show distribution of Fault_Type classes
fig = px.bar(train_df['Fault_Type'].value_counts(), 
             labels={'value': 'Count', 'index': 'Fault Type'},
             title='Distribution of Fault_Type Classes',
             template='plotly_white')
fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig)

Fault_Type_counts = train_df['Fault_Type'].value_counts()
fig = px.pie(values=Fault_Type_counts, 
             names=Fault_Type_counts.index,
             title='Distribution of Fault_Type Classes',
             template='plotly_white')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

numerical_features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
                      'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
                      'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
                      'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
                      'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
                      'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
                      'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']

# Assuming train_df is already defined

# Set the figure size and arrange plots horizontally in pairs
fig, axes = plt.subplots(nrows=(len(numerical_features) + 2) // 3, ncols=3, figsize=(30, 40))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Loop through the selected columns and create histograms with density
for i, col in enumerate(numerical_features):
    sns.histplot(data=train_df, x=col, hue='Fault_Type', multiple="stack", bins=20, kde=True, palette='viridis', ax=axes[i])
    axes[i].set_title(f'Histogram with Density for {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

plt.tight_layout()

# Display the plot using Streamlit
st.pyplot(fig)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

numerical_features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
                      'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
                      'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
                      'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
                      'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
                      'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
                      'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']

# Assuming train_df is already defined

# Create an empty list to store histogram traces
hist_traces = []

# Loop through the selected columns and create histogram traces
for col in numerical_features:
    for fault_type in train_df['Fault_Type'].unique():
        data = train_df[train_df['Fault_Type'] == fault_type][col]
        hist_trace = go.Histogram(x=data, name=fault_type, opacity=0.5)
        hist_traces.append(hist_trace)

# Create layout for the plot
layout = go.Layout(
    barmode='overlay',
    title='Histograms with Density for Numerical Features',
    xaxis=dict(title='Value'),
    yaxis=dict(title='Density'),
    showlegend=True
)

# Plot histograms using Plotly
fig = go.Figure(data=hist_traces, layout=layout)
st.plotly_chart(fig)

# Melt the DataFrame into long-form data
melted_df = pd.melt(train_df, var_name='Column', value_name='Value')

# Plot box plot using Plotly Express
fig = px.box(melted_df, x='Column', y='Value', title='Box Plot of Numerical Columns')
fig.update_layout(xaxis={'tickangle': 90}, yaxis_title='Values')
st.plotly_chart(fig)

