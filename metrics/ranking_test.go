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

func ExampleROCAUCScore() {
	// example adapted from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L256
	Ytrue := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	Yscores := mat.NewDense(4, 1, []float64{.1, .4, .35, .8})

	fmt.Println("auc:", ROCAUCScore(Ytrue, Yscores, "", nil))
	// Output:
	// auc: 0.75

}

func ExamplePrecisionRecallCurve() {
	// example adapted from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L423
	Ytrue := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	Yscores := mat.NewDense(4, 1, []float64{.1, .4, .35, .8})
	precision, recall, thresholds := PrecisionRecallCurve(Ytrue, Yscores, 1, nil)
	fmt.Printf("precision: %.3g\n", precision)
	fmt.Printf("recall: %.3g\n", recall)
	fmt.Printf("thresholds: %.3g\n", thresholds)
	// Output:
	// precision: [0.667 0.5 1 1]
	// recall: [1 0.5 0.5 0]
	// thresholds: [0.35 0.4 0.8]
}

func ExampleAveragePrecisionScore() {
	// adapted from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L180
	Ytrue := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	Yscores := mat.NewDense(4, 1, []float64{.1, .4, .35, .8})
	fmt.Printf("AveragePrecisionScore : %.2f\n", AveragePrecisionScore(Ytrue, Yscores, "macro", nil))

	// import numpy as np
	// from sklearn.metrics import average_precision_score
	// y_true = np.array([[0,0], [0,0], [1,1], [1,1]])
	// y_scores = np.array([[0.1,.5],[ 0.4,.4], [0.35,.35], [0.8,.4]])
	// average_precision_score(y_true, y_scores,"macro"),average_precision_score(y_true, y_scores,"micro")
	/*
		np.average( [0.83333333 0.41666667] , None )
		np.average( [0.63571429] , None )
		(0.625, 0.6357142857142857)
	*/

	//FIXME
	Ytrue = mat.NewDense(4, 2, []float64{0, 0, 0, 0, 1, 1, 1, 1})
	Yscores = mat.NewDense(4, 2, []float64{.1, .5, .4, .4, .35, .35, .8, .4})
	fmt.Printf("AveragePrecisionScore macro: %.3f\n", AveragePrecisionScore(Ytrue, Yscores, "macro", nil))
	fmt.Printf("AveragePrecisionScore micro: %.3f\n", AveragePrecisionScore(Ytrue, Yscores, "micro", []float64{1, 1, 1, 1}))
	// Output:
	// AveragePrecisionScore : 0.83
	// AveragePrecisionScore macro: 0.625
	// AveragePrecisionScore micro: 0.636

}
