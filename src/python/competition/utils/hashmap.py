import IPython

class HashMap():

    def __init__(self):
        self.d = {}
        return

    def fit(X, y):
        X = X.toarray()
        for state, action in zip(X, y):
            state = tuple(state)
            self.d[state] = action
        return self

    def predict(X):
        X = X.toarray()
        arr = [self.d[tuple(state)] for state in X]
        if len(arr) == 1:
            return arr[0]
        return arr