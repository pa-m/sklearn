# sklearn

Partial port of scikit-learn to [go](http://golang.org)

[![Build Status](https://travis-ci.org/pa-m/sklearn.svg?branch=master)](https://travis-ci.org/pa-m/sklearn)
[![Code Coverage](https://codecov.io/gh/pa-m/sklearn/branch/master/graph/badge.svg)](https://codecov.io/gh/pa-m/sklearn)
[![Go Report Card](https://goreportcard.com/badge/github.com/pa-m/sklearn)](https://goreportcard.com/report/github.com/pa-m/sklearn)
[![GoDoc](https://godoc.org/github.com/pa-m/sklearn?status.svg)](https://godoc.org/github.com/pa-m/sklearn)


## Examples
### cluster
[DBSCAN](https://godoc.org/github.com/pa-m/sklearn/cluster#example-DBSCAN) [KMeans](https://godoc.org/github.com/pa-m/sklearn/cluster#example-KMeans) 
### datasets
[LoadIris](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadIris) [LoadBreastCancer](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadBreastCancer) [LoadDiabetes](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadDiabetes) [LoadBoston](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadBoston) [LoadExamScore](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadExamScore) [LoadMicroChipTest](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadMicroChipTest) [LoadMnist](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadMnist) [LoadMnistWeights](https://godoc.org/github.com/pa-m/sklearn/datasets#example-LoadMnistWeights) [MakeRegression](https://godoc.org/github.com/pa-m/sklearn/datasets#example-MakeRegression) [MakeBlobs](https://godoc.org/github.com/pa-m/sklearn/datasets#example-MakeBlobs) 
### interpolate
[CubicSpline](https://godoc.org/github.com/pa-m/sklearn/interpolate#example-CubicSpline) [Interp1d](https://godoc.org/github.com/pa-m/sklearn/interpolate#example-Interp1d) [Interp2d](https://godoc.org/github.com/pa-m/sklearn/interpolate#example-Interp2d) 
### linear_model
[LinearRegression](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-LinearRegression) [BayesianRidge](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-BayesianRidge) [MultiTaskElasticNet](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-MultiTaskElasticNet) [MultiTaskLasso](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-MultiTaskLasso) [ElasticNet](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-ElasticNet) [Lasso](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-Lasso) [LassoPath](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-LassoPath) [LogisticRegression](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-LogisticRegression) [Ridge](https://godoc.org/github.com/pa-m/sklearn/linear_model#example-Ridge) 
### metrics
[AccuracyScore](https://godoc.org/github.com/pa-m/sklearn/metrics#example-AccuracyScore) [ConfusionMatrix](https://godoc.org/github.com/pa-m/sklearn/metrics#example-ConfusionMatrix) [PrecisionScore](https://godoc.org/github.com/pa-m/sklearn/metrics#example-PrecisionScore) [RecallScore](https://godoc.org/github.com/pa-m/sklearn/metrics#example-RecallScore) [F1Score](https://godoc.org/github.com/pa-m/sklearn/metrics#example-F1Score) [FBetaScore](https://godoc.org/github.com/pa-m/sklearn/metrics#example-FBetaScore) [PrecisionRecallFScoreSupport](https://godoc.org/github.com/pa-m/sklearn/metrics#example-PrecisionRecallFScoreSupport) [ROCCurve](https://godoc.org/github.com/pa-m/sklearn/metrics#example-ROCCurve) [R2Score](https://godoc.org/github.com/pa-m/sklearn/metrics#example-R2Score) 
### neighbors
[KNeighborsClassifier](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-KNeighborsClassifier) [MinkowskiDistance](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-MinkowskiDistance) [EuclideanDistance](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-EuclideanDistance) [KDTree](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-KDTree) [NearestCentroid](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-NearestCentroid) [KNeighborsRegressor](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-KNeighborsRegressor) [NearestNeighbors](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-NearestNeighbors) [NearestNeighbors_KNeighborsGraph](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-NearestNeighbors_KNeighborsGraph) [NearestNeighbors_Tree](https://godoc.org/github.com/pa-m/sklearn/neighbors#example-NearestNeighbors_Tree) 
### neural_network
[MLPClassifier](https://godoc.org/github.com/pa-m/sklearn/neural_network#example-MLPClassifier) 
### pipeline
[Pipeline](https://godoc.org/github.com/pa-m/sklearn/pipeline#example-Pipeline) 
### preprocessing
[MinMaxScaler](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-MinMaxScaler) [StandardScaler](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-StandardScaler) [AddDummyFeature](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-AddDummyFeature) [OneHotEncoder](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-OneHotEncoder) [Shuffler](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-Shuffler) [MaxAbsScaler](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-MaxAbsScaler) [Binarizer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-Binarizer) [Normalizer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-Normalizer) [Scale](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-Scale) [KernelCenterer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-KernelCenterer) [FunctionTransformer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-FunctionTransformer) [Imputer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-Imputer) [LabelBinarizer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-LabelBinarizer) [MultiLabelBinarizer](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-MultiLabelBinarizer) [PCA](https://godoc.org/github.com/pa-m/sklearn/preprocessing#example-PCA) 


All of this is 
- a personal project to get a deeper understanding of how all of this magic works
- a recent work still in progress, subject to refactoring, so interfaces may change, especially args to NewXXX
- processed with gofmt, golint, go vet
- unit tested but coverage should reach 90%
- underdocumented but python sklearn documentation should be sufficient

Many thanks to gonum and scikit-learn contributors

PRs are welcome

