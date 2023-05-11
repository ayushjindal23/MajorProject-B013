from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

model=RandomForestClassifier()
model.fit(X_train,y_train)
