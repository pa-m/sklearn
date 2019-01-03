package metrics

import (
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"

	"gonum.org/v1/gonum/mat"
)

func averageBinaryScore(binaryMetric func(Ytrue, Yscore *mat.Dense, sampleWeight []float64) float64,
	Ytrue, Yscore *mat.Dense, average string, sampleWeight []float64) float64 {
	m, n := Yscore.Dims()
	yType := typeOfTarget(Ytrue)
	if yType != "binary" && yType != "multilabel-indicator" {
		fmt.Printf("averageBinaryScore: %s format is unsupported\n", yType)
		return 0.
	}
	if yType == "binary" {
		return binaryMetric(Ytrue, Yscore, sampleWeight)
	}
	notAverageAxis := 1
	scoreWeight := sampleWeight
	var averageWeight []float64
	yt, ys := Ytrue, Yscore
	ytmat, ysmat := yt.RawMatrix(), ys.RawMatrix()

	switch average {
	case "micro":
		if scoreWeight != nil {
			tmp := make([]float64, m*n)
			for i := 0; i < n; i++ {
				copy(tmp[i*m:(i+1)*m], scoreWeight)
			}
			scoreWeight = tmp
		}
		yt, ys = mat.NewDense(m*n, 1, nil), mat.NewDense(m*n, 1, nil)
		ytmat, ysmat = yt.RawMatrix(), ys.RawMatrix()
		for j := 0; j < m; j++ {
			for i := 0; i < n; i++ {
				ytmat.Data[j*n+i] = Ytrue.At(j, i)
				ysmat.Data[j*n+i] = Yscore.At(j, i)
			}
		}
	case "weighted":
		averageWeight = make([]float64, n)
		for j, jyt := 0, 0; j < ytmat.Rows; j, jyt = j+1, jyt+ytmat.Stride {
			for i, v := range ytmat.Data[jyt : jyt+ytmat.Cols] {
				if v == 1. {
					if scoreWeight == nil {
						averageWeight[i] += v
					} else {
						averageWeight[i] += scoreWeight[j]
					}

				}
			}
		}
		if floats.Sum(averageWeight) == 0. {
			return 0.
		}
	case "samples":
		// # swap average_weight <-> score_weight
		averageWeight = scoreWeight
		scoreWeight = nil
		notAverageAxis = 0

	case "macro", "":
	default:
		fmt.Println("averageBinaryScore: unsupported average:", average)
	}
	ytshape := [2]int{ytmat.Rows, ytmat.Cols}
	NClasses := ytshape[notAverageAxis]
	scores := make([]float64, NClasses)
	for c := 0; c < NClasses; c++ {
		scores[c] = binaryMetric(
			yt.Slice(0, ytmat.Rows, c, c+1).(*mat.Dense),
			ys.Slice(0, ytmat.Rows, c, c+1).(*mat.Dense),
			scoreWeight,
		)
		//fmt.Printf("%s %g\n", base.MatStr(yt.Slice(0, ytmat.Rows, c, c+1), ys.Slice(0, ytmat.Rows, c, c+1)), scores[c])

	}
	//fmt.Printf("scores %.6f\n", scores)
	return stat.Mean(scores, averageWeight)
}

func typeOfTarget(Y *mat.Dense) string {
	Ymat := Y.RawMatrix()
	binary, mli := Ymat.Cols == 1, true
	for i := 0; i < Ymat.Cols; i++ {
		lo, hi, v := Ymat.Data[i], Ymat.Data[i], 0.
		for jY := 0; jY < Ymat.Rows*Ymat.Stride; jY = jY + Ymat.Stride {
			v = Ymat.Data[jY+i]
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		for jY := 0; jY < Ymat.Rows*Ymat.Stride; jY = jY + Ymat.Stride {
			v = Ymat.Data[jY+i]
			if (binary || mli) && v != lo && v != hi {
				binary = false
				mli = false
			}

		}
	}
	switch {
	case binary:
		return "binary"
	case mli:
		return "multilabel-indicator"
	default:
		return "unknown"
	}
}
