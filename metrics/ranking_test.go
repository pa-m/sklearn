package metrics

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleROCCurve() {
	// example adapted from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L453
	Y := mat.NewDense(4, 1, []float64{1, 1, 2, 2})
	scores := mat.NewDense(4, 1, []float64{.1, .4, .35, .8})
	fpr, tpr, thresholds := ROCCurve(Y, scores, 2., nil)
	fmt.Println("fpr:", fpr)
	fmt.Println("tpr:", tpr)
	fmt.Println("thresholds:", thresholds)
	// Output:
	// fpr: [0 0.5 0.5 1]
	// tpr: [0.5 0.5 1 1]
	// thresholds: [0.8 0.4 0.35 0.1]
}

func ExampleAUC() {
	// example adapted from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L453
	Y := mat.NewDense(4, 1, []float64{1, 1, 2, 2})
	scores := mat.NewDense(4, 1, []float64{.1, .4, .35, .8})
	fpr, tpr, _ := ROCCurve(Y, scores, 2., nil)
	fmt.Println("auc:", AUC(fpr, tpr))
	// Output:
	// auc: 0.75
}
