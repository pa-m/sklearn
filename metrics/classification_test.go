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
	fmt.Printf("%.2f", PrecisionScore(Ytrue, Ypred, "micro"))
	// Output:
	// 0.33
}

func ExampleRecallScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", RecallScore(Ytrue, Ypred, "micro"))
	// Output:
	// 0.33
}

func ExampleF1Score() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", F1Score(Ytrue, Ypred, "micro"))
	// Output:
	// 0.33
}

func ExampleFBetaScore() {
	Ytrue, Ypred := mat.NewDense(6, 1, []float64{0, 1, 2, 0, 1, 2}), mat.NewDense(6, 1, []float64{0, 2, 1, 0, 0, 1})
	fmt.Printf("%.2f", FBetaScore(Ytrue, Ypred, "micro", .5))
	// Output:
	// 0.33
}
