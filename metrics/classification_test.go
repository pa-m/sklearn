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

// >>> y_true = [0, 1, 2, 0, 1, 2]
//     >>> y_pred = [0, 2, 1, 0, 0, 1]
//     >>> precision_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
//     0.22...

func ExemplePrecisionScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", PrecisionScore(Ytrue, Ypred))
	// Output:
	// 0.22
}

func ExempleRecallScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", RecallScore(Ytrue, Ypred))
	// Output:
	// 0.33
}

func ExempleF1Score() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", F1Score(Ytrue, Ypred))
	// Output:
	// 0.26
}

func ExempleFBetaScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", FBetaScore(Ytrue, Ypred, .5))
	// Output:
	// 0.23
}
