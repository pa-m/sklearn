package metrics

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"

	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func binaryClfCurve(Ytrue, Yscore *mat.Dense, posLabel float64, sampleWeight []float64) (fps, tps, thresholds []float64) {
	m, n := Ytrue.Dims()
	if n > 1 {
		fmt.Println("Warning: ROCCurve: only first output will be used")
	}
	idx := make([]int, 0) //desc_score_indices
	for i := 0; i < m; i++ {
		idx = append(idx, i)
	}
	higherscore := func(i, j int) bool { return Yscore.At(idx[i], 0) > Yscore.At(idx[j], 0) }
	sort.Slice(idx, higherscore)
	descScoreIndices := idx
	distinctValueIndices := make([]int, 0)
	for ii := 0; ii < len(descScoreIndices); ii++ {
		if ii == 0 || Yscore.At(descScoreIndices[ii], 0) < Yscore.At(descScoreIndices[ii-1], 0) {
			distinctValueIndices = append(distinctValueIndices, descScoreIndices[ii])
		}
	}
	thresholdIdxs := distinctValueIndices
	tpw, fpw, w := 0., 0., 1.
	for _, i := range thresholdIdxs {
		if sampleWeight != nil {
			w = sampleWeight[i]
		}
		tp := Ytrue.At(i, 0) == posLabel
		if tp {
			tpw += w
		} else {
			fpw += w
		}
		tps = append(tps, tpw)
		fps = append(fps, fpw)
		thresholds = append(thresholds, Yscore.At(i, 0))
	}
	return
}

// ROCCurve Compute Receiver operating characteristic (ROC)
// y_true : array, shape = [n_samples]
// True binary labels in range {0, 1} or {-1, 1}.  If labels are not
// binary, pos_label should be explicitly given.
// y_score : array, shape = [n_samples]
// Target scores, can either be probability estimates of the positive
// class, confidence values, or non-thresholded measure of decisions
// (as returned by "decision_function" on some classifiers).
// pos_label : int or str, default=None
// Label considered as positive and others are considered negative.
// sample_weight : array-like of shape = [n_samples], optional
// Sample weights.
func ROCCurve(Ytrue, Yscore *mat.Dense, posLabel float64, sampleWeight []float64) (fpr, tpr, thresholds []float64) {
	var tps, fps []float64
	fps, tps, thresholds = binaryClfCurve(Ytrue, Yscore, posLabel, sampleWeight)
	if len(tps) == 0 || fps[0] != 0. {
		// Add an extra threshold position if necessary
		fps = append([]float64{0.}, fps...)
		tps = append([]float64{0.}, tps...)
		thresholds = append([]float64{thresholds[0] + 1.}, thresholds...)
	}
	fpr = fps
	tpr = tps

	fpmax := fps[len(fps)-1]
	if fpmax <= 0. {
		fmt.Println("No negative samples in y_true, false positive value should be meaningless")
		for i := range fpr {
			fpr[i] = math.NaN()
		}
	} else {

		floats.Scale(1./fpmax, fpr)
	}
	tpmax := tps[len(tps)-1]
	if tpmax <= 0 {
		fmt.Println("No positive samples in y_true, true positive value should be meaningless")
		for i := range tpr {
			tpr[i] = math.NaN()
		}
	} else {
		floats.Scale(1./tpmax, tps)
	}
	return
}

// AUC Compute Area Under the Curve (AUC) using the trapezoidal rule
func AUC(fpr, tpr []float64) float64 {
	auc := 0.
	if !sort.Float64sAreSorted(fpr) {
		fmt.Println("AUC: tpr is not sorted")
	}
	xp, yp := 0., 0.
	for i := range fpr {
		x, y := fpr[i], tpr[i]
		auc += (x - xp) * (y + yp) / 2.
		xp, yp = x, y
	}
	return auc
}

// ROCAUCScore compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
// y_true : array, shape = [n_samples] or [n_samples, n_classes]
// True binary labels in binary label indicators.
// y_score : array, shape = [n_samples] or [n_samples, n_classes]
// Target scores, can either be probability estimates of the positive
// class, confidence values, or non-thresholded measure of decisions
// (as returned by "decision_function" on some classifiers).
// average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
// If ``None``, the scores for each class are returned. Otherwise,
// this determines the type of averaging performed on the data:
// ``'micro'``:
// 	Calculate metrics globally by considering each element of the label
// 	indicator matrix as a label.
// ``'macro'``:
// 	Calculate metrics for each label, and find their unweighted
// 	mean.  This does not take label imbalance into account.
// ``'weighted'``:
// 	Calculate metrics for each label, and find their average, weighted
// 	by support (the number of true instances for each label).
// ``'samples'``:
// 	Calculate metrics for each instance, and find their average.
// sample_weight : array-like of shape = [n_samples], optional
// Sample weights.
// Returns auc : float
func ROCAUCScore(Ytrue, Yscore *mat.Dense, average string, sampleWeight []float64) float64 {
	NSamples, NCols := Ytrue.Dims()
	var aucs []float64
	if NCols == 1 {
		le := preprocessing.NewLabelEncoder()
		le.FitTransform(nil, Ytrue)
		output := 0
		NClasses := len(le.Classes[output])
		aucs = make([]float64, NClasses-1)
		for icl, posLabel := range le.Classes[output][1:] {
			fpr, tpr, _ := ROCCurve(Ytrue, Yscore, posLabel, sampleWeight)
			aucs[icl] = AUC(fpr, tpr)
		}
		switch average {
		case "micro", "weighted":
			return stat.Mean(aucs, le.Support[output][1:])
		default: //macro
			return floats.Sum(aucs) / float64(len(aucs))
		}
	} else {
		aucs = make([]float64, NCols)
		support := make([]float64, NCols)
		for icl := range aucs {
			fpr, tpr, _ := ROCCurve(
				Ytrue.Slice(0, NSamples, icl, icl+1).(*mat.Dense),
				Yscore.Slice(0, NSamples, icl, icl+1).(*mat.Dense),
				1,
				sampleWeight,
			)
			for i := 0; i < NSamples; i++ {
				if Ytrue.At(i, icl) == 1. {
					support[icl] += 1.
				}
			}
			aucs[icl] = AUC(fpr, tpr)
		}
		switch average {
		case "micro", "weighted":
			return stat.Mean(aucs, support)
		default: //macro
			return floats.Sum(aucs) / float64(len(aucs))
		}
	}
}

// PrecisionRecallCurve compute precision-recall pairs for different probability thresholds
//     Note: this implementation is restricted to the binary classification task.
//     The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
//     true positives and ``fp`` the number of false positives. The precision is
//     intuitively the ability of the classifier not to label as positive a sample
//     that is negative.
//     The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
//     true positives and ``fn`` the number of false negatives. The recall is
//     intuitively the ability of the classifier to find all the positive samples.
//     The last precision and recall values are 1. and 0. respectively and do not
//     have a corresponding threshold.  This ensures that the graph starts on the
//     x axis.
//     Parameters
//     y_true : array, shape = [n_samples]
//         True targets of binary classification in range {-1, 1} or {0, 1}.
//     probas_pred : array, shape = [n_samples]
//         Estimated probabilities or decision function.
//     pos_label : int or str, default=None
//         The label of the positive class
//     sample_weight : array-like of shape = [n_samples], optional
//         Sample weights.
//     Returns
//     precision : array, shape = [n_thresholds + 1]
//         Precision values such that element i is the precision of
//         predictions with score >= thresholds[i] and the last element is 1.
//     recall : array, shape = [n_thresholds + 1]
//         Decreasing recall values such that element i is the recall of
//         predictions with score >= thresholds[i] and the last element is 0.
//     thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
//         Increasing thresholds on the decision function used to compute
//         precision and recall.
func PrecisionRecallCurve(Ytrue, ProbasPred *mat.Dense, posLabel float64, sampleWeight []float64) (precision, recall, thresholds []float64) {
	var tps, fps []float64
	fps, tps, thresholds = binaryClfCurve(Ytrue, ProbasPred, posLabel, sampleWeight)
	lentps := len(tps)
	precision = make([]float64, lentps)
	recall = make([]float64, lentps)
	for i := range tps {
		precision[i] = tps[i] / (tps[i] + fps[i])
		recall[i] = tps[i] / tps[lentps-1]
	}
	// # stop when full recall attained
	// # and reverse the outputs so recall is decreasing
	lastInd := sort.SearchFloat64s(tps, tps[lentps-1])
	reverseAndAppend := func(a []float64, l int, v ...float64) []float64 {
		for i := 0; i < l/2; i++ {
			j := l - 1 - i
			a[i], a[j] = a[j], a[i]
		}
		for j := range v {
			a[l+j] = v[j]
		}
		return a[0 : l+len(v)]
	}
	precision = reverseAndAppend(precision, lastInd+1, 1.)
	recall = reverseAndAppend(recall, lastInd+1, 0.)
	thresholds = reverseAndAppend(thresholds, lastInd+1)
	return
}
