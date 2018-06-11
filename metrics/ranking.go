package metrics

import (
	"fmt"
	"math"
	"sort"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func binaryClfScore(Ytrue, Yscore *mat.Dense, posLabel float64, sampleWeight []float64) (fps, tps, thresholds []float64) {
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
	tpw, fpw, w := 0., 0., 1.
	for _, i := range distinctValueIndices {
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
func ROCCurve(Ytrue, Yscore *mat.Dense, posLabel float64, sampleWeight []float64) (fpr, tpr, thresholds []float64) {
	var tps, fps []float64
	fps, tps, thresholds = binaryClfScore(Ytrue, Yscore, posLabel, sampleWeight)
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
