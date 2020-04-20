from sklearn.svm import SVC

xs = [[-1, -1], [-2, -1], [1, 1], [2, 1]]
ys = [1, 1, -1, -1]

m = SVC(kernel='linear', C=10)
m.fit(xs, ys)
print(m.dual_coef_, m.coef_, m.intercept_)
