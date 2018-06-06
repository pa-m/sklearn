package metrics

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleAccuracyScore() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	var nilDense *mat.Dense
	normalize, sampleWeight := true, nilDense
	Ypred, Ytrue := mat.NewDense(4, 1, []float64{0, 2, 1, 3}), mat.NewDense(4, 1, []float64{0, 1, 2, 3})
	fmt.Println(AccuracyScore(Ytrue, Ypred, normalize, sampleWeight))
	fmt.Println(AccuracyScore(mat.NewDense(2, 2, []float64{0, 1, 1, 1}), mat.NewDense(2, 2, []float64{1, 1, 1, 1}), normalize, sampleWeight))
	// Output:
	// 0.5
	// 0.5
}

func ExampleConfusionMatrix() {
	// adapted from example in http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
	YTrue := mat.NewDense(6, 1, []float64{2, 0, 2, 2, 0, 1})
	YPred := mat.NewDense(6, 1, []float64{0, 0, 2, 2, 0, 2})
	fmt.Println(mat.Formatted(ConfusionMatrix(YTrue, YPred, nil)))
	// Output:
	// ⎡2  0  0⎤
	// ⎢0  0  1⎥
	// ⎣1  0  2⎦
}

func ExamplePrecisionScore() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	var sampleWeight []float64
	fmt.Printf("%.2f\n", PrecisionScore(Ytrue, Ypred, "macro", sampleWeight))
	fmt.Printf("%.2f\n", PrecisionScore(Ytrue, Ypred, "micro", sampleWeight))
	fmt.Printf("%.2f\n", PrecisionScore(Ytrue, Ypred, "weighted", sampleWeight))
	// Output:
	// 0.22
	// 0.33
	// 0.22
}

func ExampleRecallScore() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	var sampleWeight []float64
	fmt.Printf("%.2f\n", RecallScore(Ytrue, Ypred, "macro", sampleWeight))
	fmt.Printf("%.2f\n", RecallScore(Ytrue, Ypred, "micro", sampleWeight))
	fmt.Printf("%.2f\n", RecallScore(Ytrue, Ypred, "weighted", sampleWeight))
	// Output:
	// 0.33
	// 0.33
	// 0.33
}

func ExampleF1Score() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	var sampleWeight []float64
	fmt.Printf("%.2f\n", F1Score(Ytrue, Ypred, "macro", sampleWeight))
	fmt.Printf("%.2f\n", F1Score(Ytrue, Ypred, "micro", sampleWeight))
	fmt.Printf("%.2f\n", F1Score(Ytrue, Ypred, "weighted", sampleWeight))
	// Output:
	// 0.27
	// 0.33
	// 0.27
}

func ExampleFBetaScore() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	var sampleWeight []float64
	fmt.Printf("%.2f\n", FBetaScore(Ytrue, Ypred, .5, "macro", sampleWeight))
	fmt.Printf("%.2f\n", FBetaScore(Ytrue, Ypred, .5, "micro", sampleWeight))
	fmt.Printf("%.2f\n", FBetaScore(Ytrue, Ypred, .5, "weighted", sampleWeight))
	// Output:
	// 0.24
	// 0.33
	// 0.24
}

func ExamplePrecisionRecallFScoreSupport() {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	var precision, recall, fscore, support float64
	// cat, dog, pig := 0., 1., 2.
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	precision, recall, fscore, support = PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, "macro", nil, nil)
	fmt.Printf("macro %.2f\n", []float64{precision, recall, fscore, support})
	precision, recall, fscore, support = PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, "micro", nil, nil)
	fmt.Printf("micro %.2f\n", []float64{precision, recall, fscore, support})
	precision, recall, fscore, support = PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, "weighted", nil, nil)
	fmt.Printf("weighted %.2f\n", []float64{precision, recall, fscore, support})

	// Output:
	// macro [0.22 0.33 0.27 0.00]
	// micro [0.33 0.33 0.33 0.00]
	// weighted [0.22 0.33 0.27 0.00]

}
