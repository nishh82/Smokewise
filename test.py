import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = {
    "Year": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"],
    "Total (Both Sexes)": [5334100, 5160800, 5006100, 4926800, 4684400, 4159800, 3830200, 3804200],
    "Total (Males)": [3040300, 2920000, 2915600, 2865800, 2710400, 2517900, 2139400, 2168500],
    "Total (Females)": [2303800, 2240800, 2090600, 2061000, 1974000, 1641900, 1690900, 1635800]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)  # Set 'Year' as index

# Plotting

# Line plot showing trends over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df)
plt.title('Number of Smokers in Canada Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Smokers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar plot comparing different age groups or genders within each year
plt.figure(figsize=(10, 6))
sns.barplot(data=df, palette="muted")
plt.title('Number of Smokers in Canada by Gender Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Smokers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Stacked bar plot comparing the distribution of smokers across age groups or genders over the years
plt.figure(figsize=(10, 6))
df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of Smokers in Canada by Gender Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Smokers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart showing the proportion of smokers by gender for a specific year
plt.figure(figsize=(8, 8))
plt.pie(df.iloc[-1], labels=df.columns, autopct='%1.1f%%', colors=['salmon', 'lightgreen', 'skyblue'])
plt.title('Proportion of Smokers in Canada by Gender (2022)')
plt.tight_layout()
plt.show()
