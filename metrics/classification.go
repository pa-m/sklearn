package metrics

import (
	"fmt"

	"github.com/gonum/stat"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

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

/*func countTPFPTNFN(Ytrue, Ypred mat.Matrix, pivot float64) (TP, FP, TN, FN float64) {
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
}*/

// PrecisionScore v https://en.wikipedia.org/wiki/F1_score
func PrecisionScore(Ytrue, Ypred *mat.Dense, average string) float64 {
	p, _, _, _ := PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, average, []string{}, nil)
	return p
}

// RecallScore v https://en.wikipedia.org/wiki/F1_score
func RecallScore(Ytrue, Ypred *mat.Dense, average string) float64 {
	_, r, _, _ := PrecisionRecallFScoreSupport(Ytrue, Ypred, 1, nil, -1, average, []string{}, nil)
	return r
}

// F1Score v https://en.wikipedia.org/wiki/F1_score
func F1Score(Ytrue, Ypred *mat.Dense, average string) float64 {

	return FBetaScore(Ytrue, Ypred, average, 1.)
}

// FBetaScore is the weighted harmonic mean of precision and recall,
//     reaching its optimal value at 1 and its worst value at 0.
//     The `beta` parameter determines the weight of precision in the combined
//     score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
//     favors recall (``beta -> 0`` considers only precision, ``beta -> inf``
//     only recall)
func FBetaScore(Ytrue, Ypred *mat.Dense, average string, beta float64) float64 {
	_, _, f, _ := PrecisionRecallFScoreSupport(Ytrue, Ypred, beta, nil, -1, average, []string{}, nil)
	return f
}

// PrecisionRecallFScoreSupport Compute precision, recall, F-measure and support for each class
// operate only on 1st Y column
func PrecisionRecallFScoreSupport(YTrue, YPred *mat.Dense, beta float64, labels []float64, posLabel int, average string, warnFor []string, sampleWeight []float64) (precision, recall, fscore, support float64) {
	type sumstype struct{ tpsum, truesum, predsum float64 }
	type prfstype struct{ p, r, f, s float64 }
	cm := ConfusionMatrix(YTrue, YPred, sampleWeight)
	cmmat := cm.RawMatrix()
	NClasses := cmmat.Rows
	sumsperclass := make([]sumstype, NClasses)
	prfsperclass := make([]prfstype, NClasses)

	prfs := func(g sumstype) (precision, recall, fscore, support float64) {
		if g.predsum > 0. {
			precision = g.tpsum / g.predsum
		}
		if g.truesum > 0. {
			recall = g.tpsum / g.truesum
		}

		beta2 := beta * beta
		if beta2*precision+recall > 0. {
			fscore = ((1 + beta2) * precision * recall /
				(beta2*precision + recall))
		}
		return
	}
	for c := range sumsperclass {
		sumsperclass[c] = sumstype{
			tpsum: cm.At(c, c), truesum: mat.Sum(cm.RowView(c)), predsum: mat.Sum(cm.ColView(c)),
		}
		p, r, f, s := prfs(sumsperclass[c])
		prfsperclass[c] = prfstype{p, r, f, s}
	}
	if posLabel >= 0 {
		if posLabel >= NClasses {
			panic(fmt.Errorf("posLabel>=NClasses %d,%d", posLabel, NClasses))
		}
		r := &prfsperclass[posLabel]
		return r.p, r.r, r.f, r.s
	}
	if average == "macro" {
		var p, r, f, s []float64
		for c := range prfsperclass {
			prfs := &prfsperclass[c]
			p = append(p, prfs.p)
			r = append(r, prfs.r)
			f = append(f, prfs.f)
			s = append(s, prfs.s)
		}

		m := prfstype{stat.Mean(p, nil), stat.Mean(r, nil), stat.Mean(f, nil), stat.Mean(s, nil)}
		return m.p, m.r, m.f, m.s
	}
	// for average=micro
	if average == "micro" {
		var g sumstype
		for _, g1 := range sumsperclass {
			g.tpsum += g1.tpsum
			g.truesum += g1.truesum
			g.predsum += g1.predsum
		}
		precision, recall, fscore, support = prfs(g)
	}
	return precision, recall, fscore, support
}

// ConfusionMatrix Compute confusion matrix to evaluate the accuracy of a classification
// operate only on 1st Y column
func ConfusionMatrix(YTrue, YPred *mat.Dense, sampleWeight []float64) *mat.Dense {
	le := preprocessing.NewLabelEncoder()
	le.Fit(nil, YTrue)
	le.PartialFit(nil, YPred)
	_, yt := le.Transform(nil, YTrue)
	_, yp := le.Transform(nil, YPred)
	//var tn, fp, fn, tp int
	NClasses := len(le.Classes[0])
	cm := mat.NewDense(NClasses, NClasses, nil)
	ytmat := yt.RawMatrix()
	ypmat := yp.RawMatrix()
	for sample, jyt, jyp := 0, 0, 0; jyt < ytmat.Rows*ytmat.Stride; sample, jyt, jyp = sample+1, jyt+ytmat.Stride, jyp+ypmat.Stride {
		for i := range ytmat.Data[jyt : jyt+ytmat.Cols] {
			r, c := int(ytmat.Data[jyt+i]), int(ypmat.Data[jyp+i])
			w := 1.
			if sampleWeight != nil {
				w = sampleWeight[sample]
			}
			cm.Set(r, c, cm.At(r, c)+w)
		}
	}
	return cm
}
