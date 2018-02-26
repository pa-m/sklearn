package metrics

import (
	"github.com/gonum/floats"
)

type float = float64

// R2Score """R^2 (coefficient of determination) regression score function.
// Best possible score is 1.0 and it can be negative (because the
// model can be arbitrarily worse). A constant model that always
// predicts the expected value of y, disregarding the input features,
// would get a R^2 score of 0.0.
// Read more in the :ref:`User Guide <r2Score>`.
// Parameters
// ----------
// yTrue : array-like of shape = (nSamples) or (nSamples, nOutputs)
//     Ground truth (correct) target values.
// yPred : array-like of shape = (nSamples) or (nSamples, nOutputs)
//     Estimated target values.
// sampleWeight : array-like of shape = (nSamples), optional
//     Sample weights.
// multioutput : string in ['rawValues', 'uniformAverage', \
// 'varianceWeighted'] or None or array-like of shape (nOutputs)
//     Defines aggregating of multiple output scores.
//     Array-like value defines weights used to average scores.
//     Default is "uniformAverage".
//     'rawValues' :
//         Returns a full set of scores in case of multioutput input.
//     'uniformAverage' :
//         Scores of all outputs are averaged with uniform weight.
//     'varianceWeighted' :
//         Scores of all outputs are averaged, weighted by the variances
//         of each individual output.
//     .. versionchanged:: 0.19
//         Default value of multioutput is 'uniformAverage'.
// Returns
// -------
// z : float or ndarray of floats
//     The R^2 score or ndarray of scores if 'multioutput' is
//     'rawValues'.
// Notes
// -----
// This is not a symmetric function.
// Unlike most other scores, R^2 score may be negative (it need not actually
// be the square of a quantity R).
// References
// ----------
// .. [1] `Wikipedia entry on the Coefficient of determination
//         <https://en.wikipedia.org/wiki/CoefficientOfDetermination>`_
// Examples
// --------
// >>> from sklearn.metrics import r2Score
// >>> yTrue = [3, -0.5, 2, 7]
// >>> yPred = [2.5, 0.0, 2, 8]
// >>> r2Score(yTrue, yPred)  # doctest: +ELLIPSIS
// 0.948...
// >>> yTrue = [[0.5, 1], [-1, 1], [7, -6]]
// >>> yPred = [[0, 2], [-1, 2], [8, -5]]
// >>> r2Score(yTrue, yPred, multioutput='varianceWeighted')
// ... # doctest: +ELLIPSIS
// 0.938...
// >>> yTrue = [1,2,3]
// >>> yPred = [1,2,3]
// >>> r2Score(yTrue, yPred)
// 1.0
// >>> yTrue = [1,2,3]
// >>> yPred = [2,2,2]
// >>> r2Score(yTrue, yPred)
// 0.0
// >>> yTrue = [1,2,3]
// >>> yPred = [3,2,1]
// >>> r2Score(yTrue, yPred)
// -3.0
// """
func R2Score(yTrue, yPred, sampleWeight []float, multioutput string) float {
	if sampleWeight == nil {
		sampleWeight = make([]float, len(yTrue), len(yTrue))
		for i := range sampleWeight {
			sampleWeight[i] = 1.
		}
	}
	numerator := 0.
	for i := range sampleWeight {
		t := yTrue[i] - yPred[i]
		numerator += sampleWeight[i] * t * t
	}
	yTrueAvg := 0.
	sampleWeightSum := 0.
	for i := range sampleWeight {
		yTrueAvg += yTrue[i] * sampleWeight[i]
		sampleWeightSum += sampleWeight[i]
	}
	yTrueAvg /= sampleWeightSum
	denominator := 0.
	for i := range sampleWeight {
		t := yTrue[i] - yTrueAvg
		denominator += sampleWeight[i] * t * t
	}
	return 1. - numerator/denominator
}

// REScore2 is the R2 score for multioutput regressions
func R2Score2(yTrue, yPred [][]float, sampleWeight []float, multioutput string) float {
	nSamples, nOutputs := len(yTrue), len(yTrue[0])
	scores := make([]float, nOutputs)
	yTrue1, yPred1 := make([]float, nSamples, nSamples), make([]float, nSamples, nSamples)
	for j := range scores {
		for i := range yTrue {
			yTrue1[i], yPred1[i] = yTrue[i][j], yPred[i][j]
		}
		scores[j] = R2Score(yTrue1, yPred1, sampleWeight, multioutput)
	}
	// TODO implement multioutput=="variance_weighted"
	return floats.Sum(scores) / float(nOutputs)
}

func meanSquaredError(yTrue, yPred, sampleWeight []float) float {
	e := 0.
	w := 0.
	for i := range yTrue {
		e1 := yTrue[i] - yPred[i]
		e1 *= e1
		if sampleWeight == nil {
			e = e + e1
			w = w + 1
		} else {
			e = e + sampleWeight[i]*e1
			w = w + sampleWeight[i]
		}
	}
	return e / float(len(yTrue))
}
