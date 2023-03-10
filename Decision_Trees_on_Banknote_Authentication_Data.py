# In [0]:
import random
from datetime import datetime

import fpdf
import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

pdf = fpdf.FPDF()

pdf.add_page()

pdf.set_font("Arial", 'B', size=32)
pdf.cell(200, 10, txt="Final Report", ln=2, align='C')
currentDateAndTime = datetime.now()
pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt=str(currentDateAndTime), ln=8, align='C')

pdf.add_page()
pdf.set_font('arial', '', 10)
ln = 1
# In[1]:
# split data to test and train data
dataInput = pd.read_csv("BankNote_Authentication.csv")
X = dataInput.transpose()[0:4].transpose()
y = dataInput['class']


def train(train_size):
    global ln
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    dot_data = export_graphviz(clf, out_file=None,
                               filled=True, rounded=True,
                               special_characters=True, feature_names=X.columns.values, class_names=['0', '1'])

    graph = graphviz.Source(dot_data)

    imageFile = f'./graphs/BankNote_Authentication_tree_for_Size_{train_size * 100}%'
    graph.render(imageFile, format='png')

    y_pred = clf.predict(X_test)
    s = train_size * 100
    acc = metrics.accuracy_score(y_test, y_pred) * 100
    pdf.cell(0, 10, txt=f"For Split Size {s}%\tThe Accuracy: {acc}%", ln=ln)
    ln += 1
    pdf.cell(0, 10, txt="Check to See the Image in \"" + imageFile + '.png\"', ln=ln)
    ln += 2

    print(f"For Split Size {s}%\tThe Accuracy: {acc}%")


# for 25%
train(0.25)
print("=========" * 20)
# Five Random Sizes
for i in range(5):
    train(round(random.random(), 4))

# In[2]
print("=========" * 20)
accuracy_history = []
set_size = [.3, .4, .5, .6, .7]


def train_five_times(train_size):
    global ln
    Accuracy = np.zeros(5)
    tree_size = np.zeros(5)
    for num in range(5):
        rand = random.randint(0, 1000000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand,
                                                            train_size=train_size)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        dot_data = export_graphviz(clf, out_file=None,
                                   filled=True, rounded=True,
                                   special_characters=True, feature_names=X.columns.values, class_names=['0', '1'])

        graph = graphviz.Source(dot_data)

        imageFile = f'./graphs/BankNote_Authentication_tree_for_Size_{train_size * 100}%'
        graph.render(imageFile, format='png')

        y_pred = clf.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)

        Accuracy[num] = acc
        tree_size[num] = clf.tree_.node_count

        pdf.cell(0, 10,
                 txt=f"In Test number {num}\tFor Split Size{train_size * 100}%\twith Random Seed {rand}\tThe Accuracy: {acc * 100}%",
                 ln=ln)
        ln += 2

        pdf.cell(0, 10, txt="Check to See the Image in \"" + imageFile + '.png\"', ln=ln)
        ln += 2
        print(
            f"In Test number {num}\tFor Split Size{train_size * 100}%\twith Random Seed {rand}\tThe Accuracy: {acc * 100}%")

    s = train_size * 100

    print("=========" * 20)
    print(
        f'\nFor Split Set_Size {s}% \t mean={Accuracy.mean()} minimum={min(Accuracy)} maximum={max(Accuracy)}')
    print(f'For the Tree sizes \t mean={tree_size.mean()} minimum={min(tree_size)} maximum={max(tree_size)}\n')
    print("=========" * 20)

    pdf.cell(0, 10, txt="=========" * 20, ln=ln)
    ln += 1
    pdf.cell(0, 10, txt=
    f'\nFor Split Set_Size {s}% \t mean={Accuracy.mean()} minimum={min(Accuracy)} maximum={max(Accuracy)}', ln=ln)
    ln += 1
    pdf.cell(0, 10,
             txt=f'For the Tree sizes \t mean={tree_size.mean()} minimum={min(tree_size)} maximum={max(tree_size)}\n',
             ln=ln)
    ln += 1
    pdf.cell(0, 10, txt="=========" * 20, ln=ln)
    ln += 1

    accuracy_history.append([s, Accuracy.mean()])
    return clf.tree_.node_count


node_count = []

for i in set_size:
    ln = 1
    pdf.add_page()
    node_count.append(train_five_times(i))


# In[3]
# Shows accuracy against training set size
def plot_rel(s, acc, title, xLab, yLab):
    # Plot the history
    plt.plot(s, acc, color='red')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid()
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    plt.title(title)
    plt.show()


accuracy_history = np.array(accuracy_history)
plot_rel(accuracy_history[:, 0], accuracy_history[:, 1], "Shows accuracy against training set size",
         "Training set Size", "Accuracy")

plot_rel(set_size, node_count, "The number of nodes in the final tree against training set size", "Training set Size",
         "Tree Size")

node_count = np.array(node_count)

print(f'For All the Trees mean={node_count.mean()} minimum={min(node_count)} maximum={max(node_count)}')

ln += 5
pdf.set_font("Arial", size=14)
pdf.cell(0, 10, txt=f'For All the Trees mean={node_count.mean()} minimum={min(node_count)} maximum={max(node_count)}',
         ln=ln)
pdf.output(f"Final_Report.pdf")
