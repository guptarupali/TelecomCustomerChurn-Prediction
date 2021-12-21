# 1. Data Preprocessing
# Step 1: Import relevant libraries:
# Standard libraries for data analysis:
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sklearn modules for data preprocessing:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.ticker as mtick

# sklearn modules for Model Selection:
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# sklearn modules for Model Evaluation & Improvement:
from sklearn.metrics import accuracy_score, classification_report, mutual_info_score

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# Step 2: Import the dataset:
df = pd.read_csv('customer-churn-data.csv')

# Step 3: Evaluate data structure

#  Retrieving data heads
print(df.head)

# Retrieving data columns
print(df.columns)

# summary of the data frame
print(df.describe())
print(df.info())
print(df.dtypes)

# Re-validate column data types and missing values:
print(df.columns.to_series().groupby(df.dtypes).groups)

# isna any
print(df.isna().any())
print(df.isnull().sum())

# Identify unique values

# Unique values in each categorical variable:
df["PaymentMethod"].nunique()
df["PaymentMethod"].unique()
df["Contract"].nunique()
df["Contract"].unique()

# Step 4: Check target variable distribution:

df["Churn"].value_counts()


# Step 5: Clean the dataset:

# Transform the column TotalCharges into a numeric data type

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].astype("float")

# Null observations of the TotalCharges column
print(df[df['TotalCharges'].isnull()])

# Drop observations with null values- use of Regex
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)


# Step 6: Take care of missing data:
#print(df.isna().any())
#print(df.dropna(inplace=True))

# data overview
#print('Rows     : ', df.shape[0])
#print('Columns  : ', df.shape[1])
#print('\nFeatures : \n', df.columns.tolist())
#print('\nMissing values :  ', df.isnull().sum().values.sum())
#print('\nUnique values :  \n', df.nunique())
#print(df.info())
#print(df.isnull().sum())

# Drop the customerID column from the dataset as this is not required

# print(df.drop(columns='customerID', inplace=True))

# Unique elements of the PaymentMethod column
print(df.PaymentMethod.unique())

# Applying Regex to remove (automatic) from payment method names
df['PaymentMethod'] = df['PaymentMethod'].str.replace(' (automatic)', '', regex=True)

# Replace values for SeniorCitizen as a categorical feature
df['SeniorCitizen'] = df['SeniorCitizen'].replace({'1': 'Yes', '0': 'No'})

# Revalidate NA’s:
print(df.isna().any())


# 2. Data Evaluation

# Step 7: Exploratory Data Analysis:

# Step 7.1. Plot histogram of numeric Columns:
df2 = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
                    'MonthlyCharges', 'TotalCharges']]
# Histogram:

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns\n', horizontalalignment="center", fontstyle="normal", fontsize=24,
             fontfamily="sans-serif")
for i in range(df2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(df2.columns.values[i])

vals = np.size(df2.iloc[:, i].unique())
if vals >= 100:
    vals = 100

plt.hist(df2.iloc[:, i], bins=vals, color='#ec838a')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Step 7.2. Analyze the distribution of categorical variables:

contract_split = df[["customerID", "Contract"]]
sectors = contract_split.groupby("Contract")
contract_split = pd.DataFrame(sectors["customerID"].count())
contract_split.rename(columns={'customerID': 'No. of customers'}, inplace=True)
ax = contract_split[["No. of customers"]].plot.bar(title='Customers by Contract Type', legend=True, table=False,
                                                   grid=False, subplots=False, figsize=(12, 7), color='#ec838a',
                                                   fontsize=15, stacked=False)
plt.ylabel('No. of Customers\n',
           horizontalalignment="center", fontstyle="normal",
           fontsize="large", fontfamily="sans-serif")
plt.xlabel('\n Contract Type',
           horizontalalignment="center", fontstyle="normal",
           fontsize="large", fontfamily="sans-serif")
plt.title('Customers by Contract Type \n',
          horizontalalignment="center", fontstyle="normal",
          fontsize="22", fontfamily="sans-serif")
plt.legend(loc='upper right', fontsize="medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
x_labels = np.array(contract_split[["No. of customers"]])


def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.0f}".format(y_value)

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)


add_value_labels(ax)

# Step 7.3 Distribution of payment method type

payment_method_split = df[["customerID", "PaymentMethod"]]
sectors = payment_method_split.groupby("PaymentMethod")
payment_method_split = pd.DataFrame(sectors["customerID"].count())
payment_method_split.rename(columns={'customerID': 'No. of customers'}, inplace=True)
ax = payment_method_split[["No. of customers"]].plot.bar(title='Customers by Payment Method', legend=True, table=False,
                                                         grid=False, subplots=False, figsize=(15, 10), color='#ec838a',
                                                         fontsize=15, stacked=False)
plt.ylabel('No. of Customers\n',
           horizontalalignment="center", fontstyle="normal",
           fontsize="large", fontfamily="sans-serif")
plt.xlabel('\n Contract Type',
           horizontalalignment="center", fontstyle="normal",
           fontsize="large", fontfamily="sans-serif")
plt.title('Customers by Payment Method \n',
          horizontalalignment="center", fontstyle="normal", fontsize="22", fontfamily="sans-serif")
plt.legend(loc='upper right', fontsize="medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
x_labels = np.array(payment_method_split[["No. of customers"]])


def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.0f}".format(y_value)

        ax.annotate(label,
                    (x_value, y_value),
                    xytext=(0, space), textcoords="offset points",
                    ha='center', va=va)


add_value_labels(ax)


# Quantitative features:
features = ['MonthlyCharges', 'tenure']
df[features].hist(figsize=(10, 4))

# Density plots
df[features].plot(kind='density', subplots=True, layout=(1, 2),
                  sharex=False, figsize=(10, 4))
x = df['MonthlyCharges']
sns.distplot(x)

# Box plot

_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[0])

# Violin plot
sns.violinplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1])



# Data Visualisation
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols].describe()
sns.set(style='whitegrid')
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)

fig, ax = plt.subplots(1, 3, figsize=(15, 3))
df[df.Churn == 'No'][num_cols].hist(bins=35, color='blue', alpha=0.5, ax=ax)
df[df.Churn == 'Yes'][num_cols].hist(bins=35, color='orange', alpha=0.7, ax=ax)
plt.legend(['No Churn', 'Churn'], shadow=True, loc=9)

# create a figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# proportion of observation of each class
prop_response = df['Churn'].value_counts(normalize=True)

# create a bar plot showing the percentage of churn
prop_response.plot(kind='bar',
                   ax=ax,
                   color=['cyan', 'salmon'])

# set title and labels
ax.set_title('Proportion of observations of the response variable',
             fontsize=18, loc='left')
ax.set_xlabel('churn',
              fontsize=14)
ax.set_ylabel('proportion of observations',
              fontsize=14)
ax.tick_params(rotation='auto')

# eliminate the frame from the plot
spine_names = ('top', 'right', 'bottom', 'left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)


    def percentage_stacked_plot(columns_to_plot, super_title):
        number_of_columns = 2
        number_of_rows = math.ceil(len(columns_to_plot) / 2)

        # create a figure
        fig = plt.figure(figsize=(12, 5 * number_of_rows))
        fig.suptitle(super_title, fontsize=22, y=.95)

        # loop to each column name to create a subplot
        for index, column in enumerate(columns_to_plot, 1):

            # create the subplot
            ax = fig.add_subplot(number_of_rows, number_of_columns, index)

            # calculate the percentage of observations of the response variable for each group of the independent variable
            # 100% stacked bar plot
            prop_by_independent = pd.crosstab(df[column], df['Churn']).apply(lambda x: x / x.sum() * 100,
                                                                                         axis=1)

            prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                     rot=0, color=['springgreen', 'salmon'])

            # set the legend in the upper right corner
            ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                      title='Churn', fancybox=True)

            # set title and labels
            ax.set_title('Proportion of observations by ' + column,
                         fontsize=16, loc='left')

            ax.tick_params(rotation='auto')

            # eliminate the frame from the plot
            spine_names = ('top', 'right', 'bottom', 'left')
            for spine_name in spine_names:
                ax.spines[spine_name].set_visible(False)
    plt.show()
# demographic column names

# demographic column names
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

# stacked plot of demographic columns
percentage_stacked_plot(demographic_columns, 'Demographic Information')

# customer account column names
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']

# stacked plot of customer account columns
percentage_stacked_plot(account_columns, 'Customer Account Information')


def histogram_plots(columns_to_plot, super_title):

    # set number of rows and number of columns

    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    fig.suptitle(super_title, fontsize=22, y=.95)

    # loop to each demographic column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # histograms for each class (normalized histogram)
        df[df['Churn'] == 'No'][column].plot(kind='hist', ax=ax, density=True,
                                             alpha=0.5, color='cyan', label='No')
        df[df['Churn'] == 'Yes'][column].plot(kind='hist', ax=ax, density=True,
                                              alpha=0.5, color='yellow', label='Yes')


        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# customer account column names
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
# histogram of costumer account columns
histogram_plots(account_columns_numeric, 'Customer Account Information')

# services column names
services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# stacked plot of services columns
percentage_stacked_plot(services_columns, 'Services Information')


# function that computes the mutual infomation score between a categorical series and the column Churn

def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, df.Churn)

# select categorial variables excluding the response variable
categorical_variables = df.select_dtypes(include=object).drop('Churn', axis=1)

# compute the mutual information score between each categorical variable and the target
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

# visualize feature importance
print(feature_importance)

df_telco_transformed = df.copy()

# label encoding (binary variables)
label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        df_telco_transformed[column] = df_telco_transformed[column].map({'Female': 1, 'Male': 0})
    else:
        df_telco_transformed[column] = df_telco_transformed[column].map({'Yes': 1, 'No': 0})
# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

# encode categorical variables with more than two levels using one-hot encoding
df_telco_transformed = pd.get_dummies(df_telco_transformed, columns=one_hot_encoding_columns)

# min-max normalization (numeric variables)
min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# scale numerical variables using min max scaler
for column in min_max_columns:
    # minimum value of the column
    min_column = df_telco_transformed[column].min()
    # maximum value of the column
    max_column = df_telco_transformed[column].max()
    # min max scaler
    df_telco_transformed[column] = (df_telco_transformed[column] - min_column) / (max_column - min_column)


# select independent variables
X = df_telco_transformed.drop(columns='Churn')

# select dependent variables
y = df_telco_transformed.loc[:, 'Churn']

# prove that the variables were selected correctly
print(X.columns)

# prove that the variables were selected correctly
print(y.name)
# split the data in training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=40, shuffle=True)


def create_models(seed=2):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    '''

    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))

    return models


# create a list with all the algorithms we are going to assess
models = create_models()

# test the accuracy of each model using default hyperparameters

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    # fit the model with the training data
    model.fit(X_train, y_train).predict(X_test)


    # make predictions with the testing data
    predictions = model.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    # append the model name and the accuracy to the lists
    results.append(accuracy)
    names.append(name)

    # print classifier accuracy
    print('Classifier: {}, Accuracy: {})'.format(name, accuracy))



# define the parameter grid
grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],
                   'max_depth': [3, 4, 5, 6],
                   'max_features': [None, 'auto', 'sqrt', 'log2'],
                   'min_samples_split': [2, 3, 4, 5]}

# define the RandomizedSearchCV class for trying different parameter combinations
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=grid_parameters,
                                   cv=5,
                                   n_iter=330,
                                   n_jobs=-1)

# fitting the model for random search
random_search.fit(X_train, y_train)

# print best parameter after tuning
print('best parameter after tuning', random_search.best_params_)

# make the predictions
random_search_predictions = random_search.predict(X_test)

# construct the confusion matrix
confusion_matrix = confusion_matrix(y_test, random_search_predictions)


# visualize the confusion matrix
print('confusion_matrix', confusion_matrix)


# print classification report

print('classification report', classification_report(y_test, random_search_predictions))
# print the accuracy of the model
print(' accuracy of the model', accuracy_score(y_test, random_search_predictions))
