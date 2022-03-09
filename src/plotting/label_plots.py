"""
This script is used to create plots for frequency of the each classes.
Idea to the plots was from here https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
  df = pd.read_csv('resources/data/generated/train.csv', index_col='im_name')
  return df

def create_texts(title, y, x):
  plt.title(title, fontsize=20)
  plt.ylabel(y, fontsize=16)
  plt.xlabel(x, fontsize=16)

def create_barlabels(ax, labels):
  rects = ax.patches
  for rect, label in zip(rects, labels):
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

def create_class_plot():
  df = read_data()
  class_categories = list(df.columns.values)
  num_images = df.iloc[:,:].sum().values
  df2 = pd.DataFrame({'labels': class_categories, 'images': num_images})
  df2 = df2.sort_values(by=['images'], ascending=False)
  ax = sns.barplot(x=df2.labels, y=df2.images)
  create_texts('Number of images in each category class','Number of Images', 'Name of labels')
  labels = df2.images
  create_barlabels(ax, labels)
  plt.show()

def create_multi_label_plot():
  df = read_data()
  multiLabel_counts = df.iloc[:,:].sum(axis=1).value_counts().iloc[:]
  image_count = multiLabel_counts.values
  ax = sns.barplot(x=multiLabel_counts.index, y=image_count)
  create_texts('Images having multiple labels','Number of Images', 'Number of labels')
  labels = image_count
  create_barlabels(ax, labels)
  plt.show()

create_class_plot()
create_multi_label_plot()