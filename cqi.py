#Importing all relevance libs
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

#import original data set 
coffee = pd.read_csv("df_arabica_clean.csv")
print(coffee.info())
print(coffee.head())
print(coffee.isnull().sum().sort_values())

coffee.drop(["Unnamed: 0", "ID"], inplace=True, axis=1)
print(coffee.columns)
print(coffee.shape)

print(coffee["Country of Origin"].value_counts())
print(coffee[["Bag Weight"]].head())

coffee["Bag Weight"] = coffee["Bag Weight"].str.replace("kg", "")
coffee["Bag Weight"] = coffee["Bag Weight"].astype("float64")
print(coffee.dtypes)

print(coffee[["Aroma", "Flavor", "Aftertaste", "Acidity", "Sweetness", "Balance", "Bag Weight", "Defects", "Moisture Percentage"]].head())

print(coffee[["Expiration", "Harvest Year", "Grading Date"]].tail())
coffee["Expiration"] = pd.to_datetime(coffee["Expiration"], dayfirst=True, format="mixed", errors="coerce")
print(coffee["Expiration"].head())
coffee["Grading Date"] = pd.to_datetime(coffee["Grading Date"], dayfirst=True, format="mixed", errors="coerce")
print(coffee["Grading Date"].head())
print(coffee.dtypes) #check it that to convert datetimes

coffee.rename(columns={"Country of Origin":"Country",
                       "Farm Name": "Farm",
                      "Lot Number":"Lot",
                      "ICO Number":"ICO",
                      "Processing Method":"Method",
                      "Moisture Percentage":"Moisture Perc"}, inplace=True)

coffee.drop(["Certification Body", "Certification Address", "Certification Contact", "Category One Defects"], axis=1, inplace=True)
print(coffee.head())
print(coffee.nunique())

coffee["Method"] = coffee["Method"].str.replace(",", " / ").str.lower()
coffee["Method"] = coffee["Method"].str.replace("-", " ").str.capitalize()
print(coffee["Method"].value_counts())

# #Time series comparing (Expiration-Grading Year)
# fig = px.line(coffee, x="Expiration", y="Grading Date", hover_data="Company")
# fig.show()

#Top 15 Most Frequent Country
country = coffee["Country"].value_counts()
top_15 = country.nlargest(15)
print(top_15)
top_15_df = pd.DataFrame({"Country":top_15.index, "Count":top_15.values})

plt.figure(figsize=(12,6))
ax = sns.barplot(data=top_15_df, x="Count", y="Country", palette="ocean")
plt.xlabel("Count")
plt.ylabel("Country")
plt.title("Top 15 Most Frequent Country")

for i, v in enumerate(top_15_df['Count']):
    ax.text(v + 0.2, i, str(v), color='red', va='center')
plt.show()


#Typos correcting and counting colors and visualize pie chart
coffee["Color"] = coffee["Color"].str.replace("yello-green", "yellow-green")
coffee["Color"] = coffee["Color"].str.replace("yellow green", "yellow-green")
coffee["Color"] = coffee["Color"].str.replace("yellow- green", "yellow-green")
coffee["Color"] = coffee["Color"].str.replace("browish-green", "brownish-green")
coffee["Color"] = coffee["Color"].str.replace("-", " & ").str.capitalize()
print(coffee["Color"].value_counts())

fig = px.pie(coffee, names="Color", title='Coffee Colors', color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()


tastes = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Sweetness", "Balance", 
           "Defects"]
for i in tastes : 
     sns.scatterplot(coffee, x=coffee[i], y=coffee["Moisture Perc"], hue=coffee["Country"])
     plt.xlabel(i)
     plt.ylabel("Moisture Perc")
     plt.title(f"Scatter Plot :{i} vs. Moisture Perc")
     plt.show()


#Aroma vs. Sweetness
fig = px.scatter(coffee, x="Acidity", y="Aroma", hover_data="Country")
fig.show()

#Colors histplot 
fig = px.histogram(coffee, x="Color", color="Acidity")
fig.show()

#Top 15 Most Frequent Company 
Company = coffee["Company"].value_counts()
top_5_company = Company.nlargest(5)
print(top_5_company)
top_15_df = pd.DataFrame({"Company":top_5_company.index, "Count":top_5_company.values})

ax1 = sns.barplot(top_15_df, x="Count", y="Company", palette="Pastel1")
plt.xlabel("Company")
plt.ylabel("Count")
plt.title("Top 5 of Companies")
for i, v in enumerate(top_15_df['Count']):
    ax.text(v + 0.2, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()


#Harvestyear details 
print(coffee["Harvest Year"].value_counts())
Order = ["2023", "2022 / 2023", "2022", "2021 / 2022 ", "2021", "2018 / 2019", "2017 / 2018 "]
sns.countplot(x=coffee["Harvest Year"], palette="tab20", order=Order)
plt.xticks(rotation=45)
plt.show()

fig = px.pie(coffee, names="Harvest Year", values="Harvest Year", color="Harvest Year", title=
             "Harvest Year of Coffees")
fig.show() 

#tastes reviewing corr :
tastes = coffee[["Aroma", "Flavor", "Aftertaste", "Acidity", "Balance", "Total Cup Points"]]
corr_matrix = tastes.corr()
print(corr_matrix)
plt.figure(figsize=(8,4))
sns.heatmap(corr_matrix, annot=True, cbar=True, fmt=".2f")
plt.title("Correlation Matrix of Tastes")
plt.show()

#Prediction Model
X = coffee[["Aroma", "Flavor", "Aftertaste", "Acidity", "Balance"]]
y = coffee[["Total Cup Points"]]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Importing the Model
# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error
print("RMSE : {}".format(mean_squared_error(y_test, y_pred)))
