from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
from config_RL import *


# df = pd.read_csv('./XAI_DATA.csv')
df = pd.read_csv('src/XAI_DATA.csv')
# XAI data classification
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1:]
print(X, y)
# Decision tree learning
clf = DecisionTreeClassifier()
# clf = DecisionTreeClassifier(
#     criterion='gini',          # 'gini' 또는 'entropy'
#     max_depth=5,               # 트리의 최대 깊이
#     min_samples_split=20,      # 노드를 분할하기 위한 최소 샘플 수
#     min_samples_leaf=10        # 잎 노드가 가지고 있어야 할 최소 샘플 수
# )
print('start_fit')
clf = clf.fit(X, y)

FEATURE_NAME = ['Mat. InvenLevel', 'Mat. DailyChange',
                'Prod. InvenLevel', 'Prod. DailyChange', 'Remaining Demand']

# XAI
# Visualize DOT format data to create graphs
# Generate data in DOT format to visualize decision trees
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=FEATURE_NAME,
                           class_names=['[0]', '[1]',
                                        '[2]', '[3]', '[4]', '[5]'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)


# you can save the graph as a PDF file or display it on the screen.
# graph.render("decision_tree_visualization")
graph.view()
