from sklearn.preprocessing import StandardScaler
from sklearn import svm

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

clf = svm.SVC()
clf.fit(X_train,y_train)
y2 = clf.predict(X_test)
