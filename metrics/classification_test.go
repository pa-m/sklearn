package metrics

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleAccuracyScore() {
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
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f\n", PrecisionScore(Ytrue, Ypred, "macro"))
	fmt.Printf("%.2f\n", PrecisionScore(Ytrue, Ypred, "micro"))
	// Output:
	// 0.22
	// 0.33
}

func ExampleRecallScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f\n", RecallScore(Ytrue, Ypred, "macro"))
	fmt.Printf("%.2f\n", RecallScore(Ytrue, Ypred, "micro"))
	// Output:
	// 0.33
	// 0.33
}

func ExampleF1Score() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f\n", F1Score(Ytrue, Ypred, "macro"))
	fmt.Printf("%.2f\n", F1Score(Ytrue, Ypred, "micro"))
	// Output:
	// 0.27
	// 0.33
}

func ExampleFBetaScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f\n", FBetaScore(Ytrue, Ypred, "macro", .5))
	fmt.Printf("%.2f\n", FBetaScore(Ytrue, Ypred, "micro", .5))
	// Output:
	// 0.24
	// 0.33
}

func ExamplePrecisionRecallFScoreSupport() {
	var precision, recall, fscore, support float64
	// cat, dog, pig := 0., 1., 2.
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	precision, recall, fscore, support = PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, "macro", nil, nil)
	fmt.Printf("macro %.2f\n", []float64{precision, recall, fscore, support})
	precision, recall, fscore, support = PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, "micro", nil, nil)
	fmt.Printf("micro %.2f\n", []float64{precision, recall, fscore, support})

	// Output:
	// macro [0.22 0.33 0.27 0.00]
	// micro [0.33 0.33 0.33 0.00]

}
