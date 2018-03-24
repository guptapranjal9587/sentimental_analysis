from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Evaluation function

def prediction(y_test,y_pred, model):
    print("With " + model +" overall accuracy = ",accuracy_score(y_test,y_pred ))
    
def evaluate(x_train, y_train, x_test, y_test):
    regr = linear_model.LogisticRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    prediction(y_test, y_pred,"Logistic Regression")
    
    clf = MLPClassifier(activation= 'tanh',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2,), random_state=1)
    clf.fit(x_train, y_train) 
    y_pred = clf.predict(x_test)
    prediction(y_test, y_pred,"Neural network")
    
    sv = svm.SVC()
    sv.fit(x_train, y_train)
    y_pred = sv.predict(x_test)
    prediction(y_test, y_pred,"SVM")


#Evaluation by naive bayes for BBOW and NORMALIZED_TF
def NaiveBays(x_train, y_train, x_test, y_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    prediction(y_test, y_pred," Naive Bayes")