# sklearn

Very partial port of scikit-learn to [go](http://golang.org)

[![Build Status](https://travis-ci.org/pa-m/sklearn.svg?branch=master)](https://travis-ci.org/pa-m/sklearn)
[![Code Coverage](https://codecov.io/gh/pa-m/sklearn/branch/master/graph/badge.svg)](https://codecov.io/gh/pa-m/sklearn)
[![Go Report Card](https://goreportcard.com/badge/github.com/pa-m/sklearn)](https://goreportcard.com/report/github.com/pa-m/sklearn)
[![GoDoc](https://godoc.org/github.com/pa-m/sklearn?status.svg)](https://godoc.org/github.com/pa-m/sklearn)


for now, ported only some estimators including

- LinearRegression
- LogisticRegression
- [bayesian ridge regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- MLPRegressor
- MLPClassifier

You'll also find

- some metrics MeanSquaredError,MeanAbsoluteError,R2Score,AccuracyScore, ...
- some preprocessing MinMaxScaler,StandardScaler,OneHotEncoder,PolynomialFeatures
- all estimators can use  following
    - solvers:  sgd,adagrad,rmsprop,adadelta,adam + all gonum/optimize methods
    - loss functions: square,cross-entropy
    - activation functions: identity,logistic,tanh,relu

All of this is 
- a personal project to get a deeper understanding of how all of this magic works
- a work in progress, subject to refactoring, so interfaces may change, especially args to NewXXX
- processed with gofmt, golint, go vet
- unit tested but coverage should reach 90%
- underdocumented but python sklearn documentation should be sufficient

Many thanks to gonum and scikit-learn contributors

PRs are welcome

Best regards