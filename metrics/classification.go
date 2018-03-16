package metrics

import "gonum.org/v1/gonum/mat"

//	"fmt"
// nn "../neural_network"

// LogLoss ...
// func LogLoss(Ytrue, Ypred mat.Matrix) float64 {
// 	return nn.LogLoss{}.Func(Ytrue, Ypred)
// }

// AccuracyScore reports (weighted) true values/nSamples
func AccuracyScore(Ytrue, Ypred mat.Matrix, normalize bool, sampleWeight *mat.Dense) float64 {
	nSamples, nOutputs := Ytrue.Dims()
	N, D := 0., 0.
	w := 1.
	for i := 0; i < nSamples; i++ {
		if sampleWeight != nil {
			w = sampleWeight.At(i, 0)
		}
		var eq = true
		for j := 0; j < nOutputs; j++ {
			yt, yp := Ytrue.At(i, j), Ypred.At(i, j)
			if yp >= .5 {
				yp = 1.
			} else {
				yp = 0.
			}
			eq = eq && yt == yp
		}
		if eq {
			N += w
		}
		D += w
	}
	if normalize {
		return N / D
	}
	return N

}

func countTPFPTNFN(Ytrue, Ypred mat.Matrix, pivot float64) (TP, FP, TN, FN float64) {
	nSamples, nOutputs := Ytrue.Dims()
	for i := 0; i < nSamples; i++ {
		for o := 0; o < nOutputs; o++ {
			if Ypred.At(i, o) >= pivot {
				if Ytrue.At(i, o) >= pivot {
					TP += 1.
				} else {
					FP += 1.
				}
			} else {
				if Ytrue.At(i, o) >= pivot {
					FN += 1.
				} else {
					TN += 1.
				}
			}
		}
	}
	return
}

// Precision v https://en.wikipedia.org/wiki/F1_score
func PrecisionScore(Ytrue, Ypred mat.Matrix) float64 {
	pivot := 0.5
	TP, FP, _, _ := countTPFPTNFN(Ytrue, Ypred, pivot)
	return TP / (TP + FP)

}

// Recall v https://en.wikipedia.org/wiki/F1_score
func RecallScore(Ytrue, Ypred mat.Matrix) float64 {
	pivot := .5
	TP, _, _, FN := countTPFPTNFN(Ytrue, Ypred, pivot)
	return TP / (TP + FN)
}

// F1Score v https://en.wikipedia.org/wiki/F1_score
func F1Score(Ytrue, Ypred mat.Matrix) float64 {
	// P, R := PrecisionScore(Ytrue, Ypred), RecallScore(Ytrue, Ypred)
	// return 2. / ((1. / R) + (1. / P))
	return FBetaScore(Ytrue, Ypred, 1.)
}

// FBetaScore is the weighted harmonic mean of precision and recall,
//     reaching its optimal value at 1 and its worst value at 0.
//     The `beta` parameter determines the weight of precision in the combined
//     score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
//     favors recall (``beta -> 0`` considers only precision, ``beta -> inf``
//     only recall)
func FBetaScore(Ytrue, Ypred mat.Matrix, beta float64) float64 {
	Beta2 := beta * beta
	pivot := 0.5
	TP, FP, _, FN := countTPFPTNFN(Ytrue, Ypred, pivot)
	return (1 + Beta2) * TP / ((1+Beta2)*TP + Beta2*FN + FP)
}
