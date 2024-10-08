import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'


dataset = pd.read_csv('./Data/all_data_cleaned.csv')
comments = list(dataset['comment_text'])

comment_lengths = [ len(comment) for comment in comments ]

cmt_length = pd.DataFrame(comment_lengths, columns=['cmt_length'])

plt.figure(figsize=(8, 6))
sns.boxplot(data=cmt_length, y='cmt_length')
plt.title('Jigsaw Civil Comment Comment Lengths')
plt.ylabel('Length of Comments (characters)')
plt.show()

Q1 = cmt_length['cmt_length'].quantile(0.25)
Q2 = cmt_length['cmt_length'].median()
Q3 = cmt_length['cmt_length'].quantile(0.75)
IQR = Q3 - Q1

print(f"Median (Q2): {Q2}")
print(f"First Quartile (Q1): {Q1}")
print(f"Third Quartile (Q3): {Q3}")
print(f"Interquartile Range (IQR): {IQR}")