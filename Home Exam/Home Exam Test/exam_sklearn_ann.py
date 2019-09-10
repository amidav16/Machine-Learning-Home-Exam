import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

in_data_train = pd.read_csv("mnist_train.csv").as_matrix()
in_data_test = pd.read_csv("mnist_test.csv").as_matrix()

n_train_rows = 60000 #Ammount of rows in mnist_train.csv dataset
n_test_rows = 10000 #Ammount of rows in mnist_test.csv dataset
n_correct_predictions = 0 #Correct prediction iterator

clf = DecisionTreeClassifier()

#Training
print("\nTraining on", n_train_rows, "rows: \n")
training_config = in_data_train[0:n_train_rows,1:]
train_label = in_data_train[0:n_train_rows,0]
clf.fit(training_config, train_label)

#Testing
print("Testing on", n_test_rows, "rows: \n")
testing_config = in_data_test[0:,1:]
actual_label = in_data_test[0:,0]
prediction = clf.predict(testing_config)

#Predicting
for i in range(0,n_test_rows):
      n_correct_predictions+=1 if prediction[i] == actual_label[i] else 0

#Print the final result
print("Accuracy: ", (n_correct_predictions/n_test_rows)*100)



