# libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# data 
df = pd.read_csv('1.CSV')

df['Year'] = df['Year'] + (df['Month']-1)/12

del df['Month']

df['Average Picture Area'] = np.log(df['Average Picture Area'])

# print(df)


# Control figure size for this notebook:
plt.rcParams['figure.figsize'] = [8, 8]


# set seaborn "whitegrid" theme
sns.set_style("darkgrid")

# use the scatterplot function
sns.scatterplot(data=df, x="Year", y="Average Picture Area", size="Number", 
                hue='Class', palette="viridis", 
                edgecolors="black", alpha=0.5, sizes=(10, 1000))

# Add titles (main and on axis)
plt.xlabel("Year")
plt.ylabel("Average Picture Area (log pixel)")

# Locate the legend outside of the plot
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=17)

# show the graph
# plt.show()

for i in range(len(df)):
    plt.annotate(df['Dataset'][i],
    xy = (df['Year'][i], df['Average Picture Area'][i]),
    xytext = (df['Year'][i], df['Average Picture Area'][i]),
    fontsize=8
    )


current_handles, current_labels = plt.gca().get_legend_handles_labels()
current_handles = current_handles[7:]+current_handles[1:6]
current_labels = current_labels[7:]+current_labels[1:6]
plt.legend(current_handles,current_labels, ncol=2, labelspacing=2.4, markerfirst=False, borderpad=1.5)


plt.savefig ('1.png', bbox_inches='tight')