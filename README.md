# Introduction

Compare the utilization of bicycles on working days and non-working days. Specifically, I want to check if working day has an effect on the number of electric cycles rented.

🔍 Data Analysis? Check it out here: [Bikes Rental](https://github.com/nawar27/Bike-Rentals/blob/main/bike_rental_2.ipynb)


# Background

## A micro-mobility service provider

![Images](https://images.unsplash.com/photo-1668922682211-bd63b9d72126?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)
*[Source Images](https://unsplash.com/photos/a-row-of-bicycles-parked-on-a-sidewalk-ZI31cV0fFuc)*

Yulu operates shared electric bicycles across Indian cities, with zones located at high-demand areas such as metro stations, bus stops, office complexes, residential neighborhoods, and corporate campuses to support seamless first-mile and last-mile transportation.

In this case, I want to check if working day has an effect on the number of electric cycles rented. This dataset comes from this [Bike rentals](https://www.kaggle.com/datasets/sreekargv/bike-rentals)


# Tools I Used

I used several key tools to analyze data:
- **Python** : I used Python as a powerful tool for data analysis and visualization.
- **Visual Studio Code** : I used Visual Studio Code to perform the analysis using Python.
- **Git & GitHub** : Essential for version control and sharing my Python notebook, track changes and collaborate on my Python analysis project.


# The Analysis

## 1. Loading data into DataFrame

Load data from a package or library containing a dataset for the purpose of exploration, analysis, or model training.

```python
# load data
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
```

```python
csv_path = 'Yulu.csv'
df = pd.read_csv(csv_path, delimiter=',')
```

First of all, we should take a look at our dataset

- Check Missing Value and handle it if any
- Check duplicates and handle it

```python
df.head()
```

| datetime            | season | holiday | workingday | weather | temp | atemp  | humidity |
|---------------------|--------|---------|------------|---------|------|--------|----------|
| 2011-01-01 00:00:00 | 1      | 0       | 0          | 1       | 9.84 | 14.395 | 81       |
| 2011-01-01 01:00:00 | 1      | 0       | 0          | 1       | 9.02 | 13.635 | 80       |
| 2011-01-01 02:00:00 | 1      | 0       | 0          | 1       | 9.02 | 13.635 | 80       |
| 2011-01-01 03:00:00 | 1      | 0       | 0          | 1       | 9.84 | 14.395 | 75       |
| 2011-01-01 04:00:00 | 1      | 0       | 0          | 1       | 9.84 | 14.395 | 75       |


- The data in the DataFrame is this.  We'll look at the dataset's initial values and organizational structure.

```python
df.describe().transpose()
```

|         |   count |      mean |       std |   min |   25% |   50% |   75% |
|:--------|--------:|----------:|----------:|------:|------:|------:|------:|
| season  | 10886   | 2.50661   | 1.11617   |     1 |     2 |     3 |     4 |
| holiday | 10886   | 0.028569  | 0.166599  |     0 |     0 |     0 |     0 |
| workingday| 10886   | 0.680875  | 0.466159  |     0 |     0 |     1 |     1 |
| weather | 10886   | 1.41843   | 0.633839  |     1 |     1 |     1 |     2 |
| temp    | 10886   | 20.2309   | 7.79159   |     0.82 |  13.94 |  20.5 |  26.24 |
| atemp   | 10886   | 23.6551   | 8.4746    |     0.76 |  16.665|  24.24 |  31.06 |
| humidity| 10886   | 61.8865   | 19.245    |     0   |  47   |  62   |  77   |
| windspeed| 10886   | 12.7994   | 8.16454   |     0   |   7.0015|  12.998|  16.9979|
| casual  | 10886   | 36.022    | 49.9605   |     0   |   4   |  17   |  49   |
| registered| 10886   | 155.552   | 151.039   |     0   |  36   | 118   | 222   |
| count   | 10886   | 191.574   | 181.144   |     1   |  42   | 145   | 284   |


- Calculates and returns a new DataFrame containing various descriptive statistics for each numerical column in DataFrame. 


```python
# number of rows and columns in dataset
print(f'rows: {df.shape[0]} \ n columns: {df.shape}[1]')
```

- we can overview all variables and datatypes in our dataset

```python
df.info()
```
**Data Information**

RangeIndex: 10886 entries, 0 to 10885

Data columns (total 12 columns):
| #   | Column      | Non-Null Count | Dtype   |
|-----|-------------|----------------|---------|
| 0   | datetime    | 10886 non-null | object  |
| 1   | season      | 10886 non-null | int64   |
| 2   | holiday     | 10886 non-null | int64   |
| 3   | workingday  | 10886 non-null | int64   |
| 4   | weather     | 10886 non-null | int64   |
| 5   | temp        | 10886 non-null | float64 |
| 6   | atemp       | 10886 non-null | float64 |
| 7   | humidity    | 10886 non-null | int64   |
| 8   | windspeed   | 10886 non-null | float64 |
| 9   | casual      | 10886 non-null | int64   |
| 10  | registered  | 10886 non-null | int64   |
| 11  | count       | 10886 non-null | int64   |
|     | dtypes      | float64(3), int64(8), object(1) |         |
  
- Checking for missing values in the dataset.

```python
df.isnull().sum()
```

```python
df.duplicated().sum()
```

- The dataset contains no missing values or duplicate entries.

## 2. Define Null & Alternate Hypothesis

First of all, stating Null hypothesis ($H_0$), alternative hypothesis ($H_1$), and significance level

$H_0$: The bike's renting in working days and non- working days are equal.
$$ H_0:\mu_A = \mu_B $$

$H_1$: The bike's renting count in working days and non- working days are not equal.
$$ H_1:\mu_A \ne \mu_B $$

The significance level (α) is set to 0.05.


### Preanalysis

Count the number of bike sharing in workingday vs holidays

```python
df_groupby = df.groupby('workingday')['count'].sum()
df_groupby
```
out:

| holiday |    |
|---------|--------:|
| 0       | 654872 |
| 1       |   1430604 |

- Count = Count of bikes rent
- The total number of bike rentals is higher on non-holidays compared to holidays.
- Bike rentals tend to be lower on holidays

```python
import matplotlib.pyplot as plt

# Data
labels = ['Holiday', 'Working Day']
counts = df_groupby

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, counts, color=['skyblue', 'lightcoral'])

# Adding values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20000, f'{yval:,}', ha='center', va='bottom')

plt.title('Bike Rentals: Working Day vs Holiday')
plt.ylabel('Number of Rentals')
plt.tight_layout()
plt.show()
```

![Bike Rentals](images\1.png)
*Bar graph visualization comparing the number of bicycle rentals between working days and holidays.*

- The chart clearly shows that bike rentals are much higher on working days compared to holidays.
- This likely reflects commuter usage, where people rent bikes to get to work or school on regular weekdays.


### Doing Analysis

- We employ the t-test rather than the z-test because, despite having large samples (both samples contain more than 30 observations), we are unaware of the population standard deviation (σ).
- We must determine whether the specified data groups have the same variance before doing the two-sample t-test.

```python
# bike rent on weekdays
data_group1 = df[df['holiday']==0]['count'].values

# bike rent on weekend
data_group2 = df[df['holiday']==1]['count'].values

# varience
np.var(data_group1), np.var(data_group2)
```
Out:

(30171.346098942427, 34040.69710674686)


Based on the values, the variances can be considered approximately equal, so we can proceed with the standard two-sample t-test.

*   To calculate two sample proportion t-test for means, we can use stats.ttest_ind
1. import library
  - from scipy import stats
2. Use function `stats.ttest_ind(a=...., b=...., equal_var=True/False)`
  - `a`: First data group
  - `b`: Second data group
  - `equal_var = True` : The standard independent two sample t-test will be conducted by taking into consideration the equal population variances.
  - `equal_var = False` : The Welch’s t-test will be conducted by not taking into consideration the equal population variances.

3. The function will be able to return 2 output, namely statistic test and p_value.

```python
from scipy import stats
result = stats.ttest_ind(a = data_group1,
                         b = data_group2,
                         equal_var=True, # don't need to use it because the default is true
                         alternative= 'greater')
```

```python
result.pvalue
```
Out:

0.28684619416355517

```python
result.statistic
```
Out:

0.5626388963477119

```python
# menentukan aturan keputusan
if result.pvalue < 0.05:
    print('Reject the null hypothesis')
else:
    print('Failed to reject the Null hypothesis')
```
Out:

Failed to reject the Null hypothesis

- Based on the statistical analysis, there is enough evidence to suggest that working days have a significant effect on the number of electric cycles rented


Degree of freedom from two-samples

```python
df_data = len(data_group1)+len(data_group2)-2
df_data
```

```python
# plot sample distribution
x = np.arange(-4, 8, 0.001)
plt.plot(x, stats.t.pdf(x, df = df_data),
         color='blue')

# plot alpha region
x_alpha = np.arange(stats.t.ppf(1-0.05, df = df_data), 4, 0.01)
y_alpha = stats.t.pdf(x_alpha, df = df_data)
plt.fill_between(x = x_alpha,
                 y1 = y_alpha,
                 facecolor = 'red',
                 alpha = 0.35, 
                 label = 'alpha')

# plot pvalue
x_pvalue = np.arange(result.statistic, 4, 0.01)
y_pvalue = stats.t.pdf(x_pvalue, df = df_data)

plt.fill_between(x = x_pvalue,
                 y1 = y_pvalue,
                 facecolor = 'green',
                 alpha = 0.35,
                 label = 'pvalue')

# plot t-crit and t-stats
plt.axvline(np.round(result.statistic, 4),
            color = 'green',
            linestyle = '--',
            label = 't-stat')

t_crit = np.round(stats.t.ppf(1-0.05, df =df_data), 4)
plt.axvline(t_crit,
            color = 'red',
            linestyle = '--',
            label = 't-crit')

plt.legend()
plt.xlabel('t')
plt.ylabel('density')

plt.title(f't Distribution Plot with df = {df_data} \n\n t-statistic = {np.round(result.statistic, 4)}, t_crit = {np.round(t_crit,4)}, alpha = 0.05')
```

![t_distribution](images\2.png)


- The calculated t-statistic (0.5626) falls to the left of the critical t-value (1.645). This means that our test statistic does not fall within the rejection region.

- The p-value (the green shaded area) is larger than the significance level (alpha = 0.05, the red shaded area). This is another way of saying that the observed data is not sufficiently unlikely under the assumption of the null hypothesis.

- Therefore, based on this analysis, we would fail to reject the null hypothesis. There is not enough statistical evidence at the 0.05 significance level to support the alternative hypothesis.

```python
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

cm = CompareMeans(d1 = DescrStatsW(data=data_group1), 
                  d2 = DescrStatsW(data=data_group2))

lower, upper = cm.tconfint_diff(alpha=0.05, 
                                alternative='two-sided', 
                                usevar='unequal')

print("Confidence Interval", ":", "[", lower, upper, "]")
```
Out:

Confidence Interval : [ -13.22789740496096 24.955580087986096 ]

- We are 95% confident that the true difference in the average number of bike rentals between holidays and workingday lies between -13.2 and 24.9 bikes.


# What I Learned
- Data Loading and Preprocessing: Importing necessary libraries and loading the dataset, followed by cleaning and preparing the data for analysis.

- Exploratory Data Analysis (EDA): Visualizing data distributions, identifying patterns, and understanding relationships between variables.

- Feature Engineering: Creating new features or modifying existing ones to improve model performance.

- Model Building and Evaluation: Applying machine learning algorithms to predict bike rentals and evaluating model performance using appropriate metrics.


# Conclusion
The p-value of 0.287 is greater than the significance level of 0.05, therefore we fail to reject the null hypothesis. This suggests that there is no statistically significant difference in the average number of bike rentals between non-holidays and holidays.

The 95% confidence interval for the difference in bike rentals between holiday and working days is from -13.2 to 24.9. Since this interval includes 0, it supports the conclusion that the difference is not statistically significant.
