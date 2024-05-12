
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import os

def read_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    result_csv_folder = os.path.join(parent_dir, "result_CSV")
    STATE_folder = os.path.join(result_csv_folder, "state")
    Data_place=os.path.join(STATE_folder,f"Train_{len(os.listdir(STATE_folder))}")
    return Data_place

print(read_path)
df = pd.read_csv(
    os.path.join(read_path(),"STATE_ACTION_REPORT_REAL_TEST.csv"))

# XAI data classification
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1:]
print(X, y)
# Decision tree learning
clf = DecisionTreeClassifier(
    criterion='gini',          # 'gini' 또는 'entropy'
    max_depth=6,               # 트리의 최대 깊이
    min_samples_split=20,      # 노드를 분할하기 위한 최소 샘플 수
    min_samples_leaf=10        # 잎 노드가 가지고 있어야 할 최소 샘플 수
)
print('start_fit')
clf = clf.fit(X, y)

FEATURE_NAME = df.columns[1:-1]
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
graph.render('decision_tree_visualization',  format='png', view=False)

