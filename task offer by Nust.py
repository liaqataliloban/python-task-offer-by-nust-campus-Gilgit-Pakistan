#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Task 1: NumPy Basics

import numpy as np

# Create a NumPy array with random integers
array_size = 5
min_value = 1
max_value = 10
random_array = np.random.randint(min_value, max_value + 1, size=array_size)

print("Random Array:")
print(random_array)

# Perform basic mathematical operations
addition_result = random_array + 5
subtraction_result = random_array - 2
multiplication_result = random_array * 3
division_result = random_array / 2

print("\nAddition Result:")
print(addition_result)

print("\nSubtraction Result:")
print(subtraction_result)

print("\nMultiplication Result:")
print(multiplication_result)

print("\nDivision Result:")
print(division_result)


# In[2]:


#Task 2: Pandas Data Analysis

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
columns = iris.feature_names

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=columns)

# Basic data analysis
# Calculate mean, median, and mode for specific columns
mean_sepal_length = df['sepal length (cm)'].mean()
median_sepal_length = df['sepal length (cm)'].median()
mode_sepal_length = df['sepal length (cm)'].mode()[0]

mean_sepal_width = df['sepal width (cm)'].mean()
median_sepal_width = df['sepal width (cm)'].median()
mode_sepal_width = df['sepal width (cm)'].mode()[0]

mean_petal_length = df['petal length (cm)'].mean()
median_petal_length = df['petal length (cm)'].median()
mode_petal_length = df['petal length (cm)'].mode()[0]

mean_petal_width = df['petal width (cm)'].mean()
median_petal_width = df['petal width (cm)'].median()
mode_petal_width = df['petal width (cm)'].mode()[0]

# Print the results
print("Statistics for Sepal Length:")
print(f"Mean: {mean_sepal_length}")
print(f"Median: {median_sepal_length}")
print(f"Mode: {mode_sepal_length}")

print("\nStatistics for Sepal Width:")
print(f"Mean: {mean_sepal_width}")
print(f"Median: {median_sepal_width}")
print(f"Mode: {mode_sepal_width}")

print("\nStatistics for Petal Length:")
print(f"Mean: {mean_petal_length}")
print(f"Median: {median_petal_length}")
print(f"Mode: {mode_petal_length}")

print("\nStatistics for Petal Width:")
print(f"Mean: {mean_petal_width}")
print(f"Median: {median_petal_width}")
print(f"Mode: {mode_petal_width}")


# In[3]:


#Task 3: Data Visualization

import matplotlib.pyplot as plt
import numpy as np

# Generate a random dataset (replace this with your own data)
data = np.random.randn(1000)  # Generating random data for demonstration purposes

# Create a histogram
plt.hist(data, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed

# Label the axes and add a title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Randomly Generated Data')

# Show the histogram
plt.show()


# In[4]:


#Task 4: NumPy Array Manipulation

import numpy as np

# Create two NumPy arrays
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])

array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])

# Concatenate them vertically (stacking rows)
vertical_concatenation = np.concatenate((array1, array2), axis=0)

# Concatenate them horizontally (stacking columns)
horizontal_concatenation = np.concatenate((array1, array2), axis=1)

# Print the original arrays and the concatenated arrays
print("Array 1:")
print(array1)

print("\nArray 2:")
print(array2)

print("\nVertically Concatenated:")
print(vertical_concatenation)

print("\nHorizontally Concatenated:")
print(horizontal_concatenation)


# In[5]:


#Task 5: Pandas Data Filtering

import pandas as pd

# Sample dataset (you can replace this with your own dataset)
data = {
    'Date': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05'],
    'Temperature (Celsius)': [25.5, 26.2, 24.8, 27.3, 23.9],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Define the date range you want to filter
start_date = '2023-09-02'
end_date = '2023-09-04'

# Filter and extract rows within the date range
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Display the filtered DataFrame
print("Filtered Data:")
print(filtered_df)


# In[6]:


#Task 6: Matplotlib Customization

import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 50)
y1 = x + np.random.randn(50)  # Data series 1
y2 = 2 * x + np.random.randn(50)  # Data series 2

# Create a scatter plot with different marker styles and colors
plt.scatter(x, y1, label='Data Series 1', color='blue', marker='o')
plt.scatter(x, y2, label='Data Series 2', color='red', marker='x')

# Add a legend
plt.legend()

# Customize labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Customization')

# Show the plot
plt.grid(True)
plt.show()


# In[7]:


#Task 7: NumPy Statistical Analysis

#Here's a sample dataset:

import numpy as np

# Sample dataset
math_scores = np.array([85, 90, 88, 92, 78, 82, 95, 88, 76, 87])
english_scores = np.array([88, 86, 90, 82, 75, 92, 89, 78, 84, 80])

#Variance and Standard Deviation for Math and English scores

# Variance
math_variance = np.var(math_scores)
english_variance = np.var(english_scores)

# Standard Deviation
math_std_deviation = np.std(math_scores)
english_std_deviation = np.std(english_scores)

print("Math Variance:", math_variance)
print("Math Standard Deviation:", math_std_deviation)
print("English Variance:", english_variance)
print("English Standard Deviation:", english_std_deviation)


#Correlation coefficient between Math and English scores


# Correlation coefficient
correlation_coefficient = np.corrcoef(math_scores, english_scores)[0, 1]

print("Correlation Coefficient:", correlation_coefficient)



# In[9]:


#Task 8: Pandas Data Grouping

import pandas as pd

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_df = pd.read_csv(url, header=None, names=column_names)

# Group the data by the "species" column
grouped = iris_df.groupby("species")

# Calculate summary statistics for each group
summary_statistics = grouped.agg({"sepal_length": ["mean", "median"],
                                   "sepal_width": ["mean", "median"],
                                   "petal_length": ["mean", "median"],
                                   "petal_width": ["mean", "median"]})

# Display the summary statistics
print(summary_statistics)


# In[10]:


#Task 9: Matplotlib Subplots

import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for improved plotting styles
import pandas as pd

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_df = pd.read_csv(url, header=None, names=column_names)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Subplot 1: Histogram of Sepal Length
sns.histplot(iris_df["sepal_length"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Histogram of Sepal Length")
axes[0, 0].set_xlabel("Sepal Length")
axes[0, 0].set_ylabel("Frequency")

# Subplot 2: Histogram of Sepal Width
sns.histplot(iris_df["sepal_width"], kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Histogram of Sepal Width")
axes[0, 1].set_xlabel("Sepal Width")
axes[0, 1].set_ylabel("Frequency")

# Subplot 3: Histogram of Petal Length
sns.histplot(iris_df["petal_length"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Histogram of Petal Length")
axes[1, 0].set_xlabel("Petal Length")
axes[1, 0].set_ylabel("Frequency")

# Subplot 4: Scatterplot of Sepal Length vs. Sepal Width
sns.scatterplot(data=iris_df, x="sepal_length", y="sepal_width", hue="species", ax=axes[1, 1])
axes[1, 1].set_title("Scatterplot of Sepal Length vs. Sepal Width")
axes[1, 1].set_xlabel("Sepal Length")
axes[1, 1].set_ylabel("Sepal Width")

# Add a title to the overall figure
fig.suptitle("Iris Dataset Analysis")

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()


# In[11]:


#Task 10: Pandas Data Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'TotalSales': np.random.randint(10000, 30000, size=12),
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing',
                 'Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing']
}

df = pd.DataFrame(data)

# Visualization 1: Bar chart to show monthly sales trends
plt.figure(figsize=(10, 5))
plt.bar(df['Month'], df['TotalSales'], color='skyblue')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trends')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Visualization 2: Pie chart to show the distribution of product categories
category_counts = df['Category'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightgreen'])
plt.axis('equal')
plt.title('Product Category Distribution')

# Visualization 3: Scatter plot to visualize the relationship between TotalSales and Month
plt.figure(figsize=(10, 5))
plt.scatter(df['Month'], df['TotalSales'], c='b', marker='o')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Relationship between Total Sales and Month')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:




