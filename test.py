from sklearn import cross_validation, datasets 

iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,train_size=.8, stratify=y)

y_test

array([0, 0, 0, 0, 2, 2, 1, 0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 1, 2, 0, 2, 2,
       1, 2, 1, 1, 0, 2, 1])