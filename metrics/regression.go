package metrics

import (
	"gonum.org/v1/gonum/mat"
)

type float = float64

type constVector struct {
	Value      float64
	Length     int
	Transposed bool
}

func (v constVector) Dims() (int, int) {
	if v.Transposed {
		return 1, v.Length
	}
	return v.Length, 1

}
func (v constVector) At(int, int) float64 { return v.Value }
func (v constVector) T() mat.Matrix       { return &constVector{v.Value, v.Length, !v.Transposed} }
func (v constVector) Len() int            { return v.Length }
func (v constVector) AtVec(int) float64   { return v.Value }

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
func R2Score(yTrue, yPred *mat.Dense, sampleWeight *mat.VecDense, multioutput string) *mat.Dense {
	nSamples, nOutputs := yTrue.Dims()
	if sampleWeight == nil {

		sampleWeight = mat.VecDenseCopyOf(constVector{1., nSamples, false})

	}
	numerator := mat.NewDense(1, nOutputs, nil)
	diff := mat.NewDense(nSamples, nOutputs, nil)
	diff.Sub(yPred, yTrue)
	diff2 := mat.NewDense(nSamples, nOutputs, nil)
	diff2.MulElem(diff, diff)
	numerator.Mul(sampleWeight.T(), diff2)

	sampleWeightSum := mat.NewDense(1, 1, nil)
	sampleWeightSum.Mul(sampleWeight.T(), constVector{1, nSamples, false})

	yTrueAvg := mat.NewDense(1, nOutputs, nil)
	yTrueAvg.Mul(sampleWeight.T(), yTrue)
	yTrueAvg.Scale(1./sampleWeightSum.At(0, 0), yTrueAvg)

	diff2.Apply(func(i int, j int, v float64) float64 {
		v = yTrue.At(i, j) - yTrueAvg.At(0, j)
		return v * v
	}, diff2)
	denominator := mat.NewDense(1, nOutputs, nil)
	denominator.Mul(sampleWeight.T(), diff2)

	r2score := mat.NewDense(1, nOutputs, nil)
	quotient := mat.NewDense(1, nOutputs, nil)
	quotient.DivElem(numerator, denominator)
	r2score.Sub(constVector{1, nOutputs, true}, quotient)
	switch multioutput {
	case "raw_values":
		return r2score
	case "variance_weighted":
		r2 := mat.NewDense(1, 1, nil)
		r2.Mul(denominator, r2score.T())
		sumden := mat.NewDense(1, 1, nil)
		sumden.Mul(denominator, constVector{1, nOutputs, false})
		r2.Scale(1./sumden.At(0, 0), r2)
		return r2
	default: // "uniform_average":
		r2 := mat.NewDense(1, 1, nil)
		r2.Mul(constVector{1. / float(nOutputs), nOutputs, true}, r2score.T())
		return r2score
	}

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
