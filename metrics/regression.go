package metrics

import (
	//	"fmt"
	"math"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

type float = float64

type constVector = base.MatConst

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
func R2Score(yTrue, yPred mat.Matrix, sampleWeight *mat.Dense, multioutput string) *mat.Dense {
	nSamples, nOutputs := yTrue.Dims()
	if sampleWeight == nil {

		sampleWeight = mat.DenseCopyOf(base.MatConst{Rows: nSamples, Columns: 1, Value: 1.})

	}
	numerator := mat.NewDense(1, nOutputs, nil)
	diff := mat.NewDense(nSamples, nOutputs, nil)
	diff.Sub(yPred, yTrue)
	diff2 := mat.NewDense(nSamples, nOutputs, nil)
	diff2.MulElem(diff, diff)
	numerator.Mul(sampleWeight.T(), diff2)

	sampleWeightSum := mat.Sum(sampleWeight)

	yTrueAvg := mat.NewDense(1, nOutputs, nil)
	yTrueAvg.Mul(sampleWeight.T(), yTrue)
	yTrueAvg.Scale(1./sampleWeightSum, yTrueAvg)

	diff2.Apply(func(i int, j int, _ float64) float64 {
		v := yTrue.At(i, j) - yTrueAvg.At(0, j)
		return v * v
	}, diff2)
	denominator := mat.NewDense(1, nOutputs, nil)
	denominator.Mul(sampleWeight.T(), diff2)

	r2score := mat.NewDense(1, nOutputs, nil)
	r2score.Apply(func(i int, j int, v float64) float64 {
		d := math.Max(denominator.At(i, j), 1e-20)
		return 1. - numerator.At(i, j)/d
	}, r2score)
	switch multioutput {
	case "raw_values":
		return r2score
	case "variance_weighted":
		r2 := mat.NewDense(1, 1, nil)
		r2.Mul(denominator, r2score.T())
		sumden := mat.Sum(denominator)
		r2.Scale(1./sumden, r2)
		return r2
	default: // "uniform_average":
		return mat.NewDense(1, 1, []float64{mat.Sum(r2score) / float64(nOutputs)})
	}

}

// MeanSquaredError regression loss
// Read more in the :ref:`User Guide <mean_squared_error>`.
// Parameters
// ----------
// y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
//     Ground truth (correct) target values.
// y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
//     Estimated target values.
// sample_weight : array-like of shape = (n_samples), optional
//     Sample weights.
// multioutput : string in ['raw_values', 'uniform_average']
//     or array-like of shape (n_outputs)
//     Defines aggregating of multiple output values.
//     Array-like value defines weights used to average errors.
//     'raw_values' :
//         Returns a full set of errors in case of multioutput input.
//     'uniform_average' :
//         Errors of all outputs are averaged with uniform weight.
// Returns
// -------
// loss : float or ndarray of floats
//     A non-negative floating point value (the best value is 0.0), or an
//     array of floating point values, one for each individual target.
func MeanSquaredError(yTrue, yPred mat.Matrix, sampleWeight *mat.Dense, multioutput string) *mat.Dense {
	nSamples, nOutputs := yTrue.Dims()
	tmp := mat.NewDense(1, nOutputs, nil)

	tmp.Apply(func(_ int, j int, v float64) float64 {
		N, D := 0., 0.
		for i := 0; i < nSamples; i++ {
			ydiff := yPred.At(i, j) - yTrue.At(i, j)
			w := 1.
			if sampleWeight != nil {
				w = sampleWeight.At(0, j)
			}
			N += w * (ydiff * ydiff)
			D += w
		}
		return N / D
	}, tmp)
	switch multioutput {
	case "raw_values":
		return tmp
	default: // "uniform_average":
		return mat.NewDense(1, 1, []float64{mat.Sum(tmp) / float64(nOutputs)})
	}
}

// MeanAbsoluteError regression loss
// Read more in the :ref:`User Guide <mean_absolute_error>`.
// Parameters
// ----------
// y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
//     Ground truth (correct) target values.
// y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
//     Estimated target values.
// sample_weight : array-like of shape = (n_samples), optional
//     Sample weights.
// multioutput : string in ['raw_values', 'uniform_average']
//     or array-like of shape (n_outputs)
//     Defines aggregating of multiple output values.
//     Array-like value defines weights used to average errors.
//     'raw_values' :
//         Returns a full set of errors in case of multioutput input.
//     'uniform_average' :
//         Errors of all outputs are averaged with uniform weight.
// Returns
// -------
// loss : float or ndarray of floats
//     If multioutput is 'raw_values', then mean absolute error is returned
//     for each output separately.
//     If multioutput is 'uniform_average' or an ndarray of weights, then the
//     weighted average of all output errors is returned.
//     MAE output is non-negative floating point. The best value is 0.0.
// Examples
// --------
// >>> from sklearn.metrics import mean_absolute_error
// >>> y_true = [3, -0.5, 2, 7]
// >>> y_pred = [2.5, 0.0, 2, 8]
// >>> mean_absolute_error(y_true, y_pred)
// 0.5
// >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
// >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
// >>> mean_absolute_error(y_true, y_pred)
// 0.75
// >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
// array([ 0.5,  1. ])
// >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
// ... # doctest: +ELLIPSIS
// 0.849...
func MeanAbsoluteError(yTrue, yPred mat.Matrix, sampleWeight *mat.Dense, multioutput string) *mat.Dense {
	nSamples, nOutputs := yTrue.Dims()
	tmp := mat.NewDense(1, nOutputs, nil)

	tmp.Apply(func(_ int, j int, v float64) float64 {
		N, D := 0., 0.
		for i := 0; i < nSamples; i++ {
			ydiff := yPred.At(i, j) - yTrue.At(i, j)
			w := 1.
			if sampleWeight != nil {
				w = sampleWeight.At(0, j)
			}
			N += w * math.Abs(ydiff)
			D += w
		}
		return N / D
	}, tmp)

	switch multioutput {
	case "raw_values":
		return tmp
	default: // "uniform_average":
		return mat.NewDense(1, 1, []float64{mat.Sum(tmp) / float64(nOutputs)})
	}
}
