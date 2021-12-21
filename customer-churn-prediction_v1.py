# *********************** Customer Attrition Prediction Project ***********************************
# **********************Data Preprocessing*********************************

# Step 1: Import relevant libraries:

# Standard libraries for data analysis:
import math
import pandas as pd
import numpy as np

# Standard libraries for data visualisation:
import matplotlib.pyplot as plt
import seaborn as sns

# Regex & time
import re
import time

# sklearn modules for data preprocessing:
from sklearn.model_selection import train_test_split

# sklearn modules for Model Selection:
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



# sklearn modules for Model Evaluation & Improvement:
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mutual_info_score

# Data Reading
from sklearn.tree import DecisionTreeClassifier

# Step 2: Import the dataset:

df = pd.read_csv('customer-churn-data.csv')


#  Data Preprocessing - Evaluate data structure:

#  Retrieving data heads
print(df.head())
#  Retrieving data shape
print(df.shape)
# Retrieving data columns
print(df.columns)


# summary of the data frame
# only 3 feature contain numerical values, rest are categorical feature
print(df.describe())
print(df.dtypes)


# summary of the data frame
print(df.info())

#*******************Data Preprocessing****************

# Data Cleaning, Missing values and data types
# Transform the column TotalCharges into a numeric data type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.isnull().sum()
# Null observations of the TotalCharges column
print(df[df['TotalCharges'].isnull()])

# Drop observations with null values- use of Regex
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
print(df.dropna(inplace=True))

# data overview
print('Rows     : ', df.shape[0])
print('Columns  : ', df.shape[1])
print('\nFeatures : \n', df.columns.tolist())
print('\nMissing values :  ', df.isnull().sum().values.sum())
print('\nUnique values :  \n', df.nunique())
print(df.info())

print(df.isnull().any())
print("\n# of Null values in 'TotalCharges`: ", df["TotalCharges"].isnull().sum())

# drop the customerID column from the dataset as this is not required
print(df.drop(columns='customerID', inplace=True))

# unique elements of the PaymentMethod column
print(df.PaymentMethod.unique())

# Applying Regex to remove (automatic) from payment method names
df['PaymentMethod'] = df['PaymentMethod'].apply(lambda x: re.sub(' (automatic)', ' ', x))

# Replace values for SeniorCitizen as a categorical feature
df['SeniorCitizen'] = df['SeniorCitizen'].replace({'1': 'Yes', '0': 'No'})

# Data Visualisation
# To analyse categorical feature distribution
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


num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols].describe()
sns.set(style='whitegrid')
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)

fig, ax = plt.subplots(1, 3, figsize=(15, 3))
df[df.Churn == 'No'][num_cols].hist(bins=35, color='blue', alpha=0.5, ax=ax)
df[df.Churn == 'Yes'][num_cols].hist(bins=35, color='orange', alpha=0.7, ax=ax)
plt.legend(['No Churn', 'Churn'], shadow=True, loc=9)

# scatterplot
_, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x="TotalCharges", y="tenure" , hue="Churn", data=df, ax=ax[0])
sns.scatterplot(x="MonthlyCharges", y="tenure", hue="Churn", data=df, ax=ax[1])

# Facegrid
facet = sns.FacetGrid(df, hue="Churn", aspect = 3)
facet.map(sns.kdeplot, "TotalCharges", shade=True)
facet.set(xlim=(0, df["TotalCharges"].max()))
facet.add_legend()

facet = sns.FacetGrid(df, hue="Churn", aspect=3)
facet.map(sns.kdeplot, "MonthlyCharges", shade=True)
facet.set(xlim=(0, df["MonthlyCharges"].max()))
facet.add_legend()

# Counter plot
_, axs = plt.subplots(1, 2, figsize=(9, 6))
plt.subplots_adjust(wspace=0.3)
ax = sns.countplot(data=df, x="SeniorCitizen", hue="Churn", ax=axs[0])
ax1 = sns.countplot(data=df, x="MultipleLines", hue="Churn", ax=axs[1])

for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 3.4,
        "{:1.2f}%".format(height / len(df), 0),
        ha="center", rotation=0
    )

for p in ax1.patches:
    height = p.get_height()
    ax1.text(
        p.get_x() + p.get_width() / 2,
        height + 3.4,
        "{:1.2f}%".format(height / len(df), 0),
        ha="center", rotation=0
    )


# Create a figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Proportion of observation of each class, normalization
prop_response = df['Churn'].value_counts(normalize=True)

# Create a bar plot showing the percentage of churn
prop_response.plot(kind='bar',
                   ax=ax,
                   color=['cyan', 'yellow'])

# Set title and labels
ax.set_title('Proportion of observations of the response variable',
             fontsize=20, loc='left')
ax.set_xlabel('Churn',
              fontsize=18)
ax.set_ylabel('Proportion of observations',
              fontsize=18)
ax.tick_params(rotation='auto')

# eliminate the frame from the plot
spine_names = ('top', 'right', 'bottom', 'left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)
    plt.show()

# unique elements of the PaymentMethod column after the modification
print(df.PaymentMethod.unique())


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
                                             alpha=0.5, color='springgreen', label='No')
        df[df['Churn'] == 'Yes'][column].plot(kind='hist', ax=ax, density=True,
                                              alpha=0.5, color='salmon', label='Yes')


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
    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('SVC', SVC(kernel='linear', random_state=0)))
    models.append(('Kernel SVM', SVC(kernel='rbf', random_state=0)))
    models.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)))
    models.append(('logistic_regression', LogisticRegression(solver='liblinear', random_state=0, class_weight='balanced')))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(criterion='entropy', random_state=0)))
    models.append(('Gaussian NB', GaussianNB()))
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
df_cm = pd.DataFrame(confusion_matrix, index=(0, 1), columns=(0, 1))
plt.figure(figsize=(28, 20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
class_names = [0, 1];
tick_marks = np.arange(len(class_names));
plt.tight_layout();
plt.title('Confusion matrix\n', y=1.1);
plt.xticks(tick_marks, class_names);
plt.yticks(tick_marks, class_names);
ax.xaxis.set_label_position("top");
plt.ylabel('Actual label\n');
plt.xlabel('Predicted label\n');
# visualize the confusion matrix
print('confusion_matrix', confusion_matrix);
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, random_search_predictions));

# print classification report

print('classification report', classification_report(y_test, random_search_predictions))
# print the accuracy of the model
print(' accuracy of the model', accuracy_score(y_test, random_search_predictions))

