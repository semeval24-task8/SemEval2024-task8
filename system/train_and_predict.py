from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import argparse
import numpy as np

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", type=str, help="Train file (csv)")
    parser.add_argument("-ts", "--test", type=str, help="Test file (csv)")
    return parser.parse_args()

def main():
  args = create_arg_parser()
  clf = MultinomialNB()
  train = np.genfromtxt(args.train, delimiter=",", skip_header=1)
  test = np.genfromtxt(args.test, delimiter=",", skip_header=1)
  # train_x = train[:, 2:]
  # train_y = train[:, 1]
  # test_x = test[:, 2:]
  # test_y = test[:, 1]
  train_x = train[:, 1:-1]
  train_y = train[:, -1]
  test_x = test[:, 1:-1]
  test_y = test[:, -1]
  clf = clf.fit(train_x, train_y)
  test_pred = clf.predict(test_x)
  print(classification_report(test_y, test_pred, digits=3))

main()