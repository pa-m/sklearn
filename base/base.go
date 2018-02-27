package base

import (
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

type float = float64

// Predicter is an interface for Predict method
type Predicter interface {
	Predict([][]float) []float
}

// Predicter2 is an interface for Predict method
type Predicter2 interface {
	Predict([][]float) [][]float
}

// RegressorMixin1 is a base for predicters. provides a Score(X,w,weights) method
type RegressorMixin1 struct{ Predicter }

func v2dense(v []float) *mat.Dense {
	if v == nil {
		return nil
	}
	return mat.NewDense(len(v), 1, v)
}
func v2vecdense(v []float) *mat.VecDense {
	if v == nil {
		return nil
	}
	return mat.NewVecDense(len(v), v)
}
func m2dense(X [][]float) *mat.Dense {
	r, c := len(X), len(X[0])
	m := mat.NewDense(r, c, nil)
	m.Apply(func(i int, j int, v float64) float64 {
		return X[i][j]
	}, m)
	return m
}

// Score returns R2Score of predicter
func (predicter *RegressorMixin1) Score(X [][]float, y, sampleWeight []float) float {
	yPred := predicter.Predict(X)
	return metrics.R2Score(v2dense(y), v2dense(yPred), v2vecdense(sampleWeight), "variance_weighted").At(0, 0)
}

// RegressorMixin2 is a base for predicters. provides a Score(X,w,weights) method
type RegressorMixin2 struct{ Predicter2 }

// Score returns R2Score of predicter
func (predicter *RegressorMixin2) Score(X, y [][]float, sampleWeight []float) float {
	yPred := predicter.Predict(X)
	return metrics.R2Score(m2dense(y), m2dense(yPred), v2vecdense(sampleWeight), "variance_weighted").At(0, 0)
}

// Shuffle shuffles X,y samples
func Shuffle(X [][]float, y []float) {
	for i := len(X) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		X[i], X[j] = X[j], X[i]
		y[i], y[j] = y[j], y[i]
	}
}

// DenseShuffle shuffles the rows of X and Y matrices
func DenseShuffle(X, Y *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Y.Dims()
	Xrowi := make([]float64, nFeatures, nFeatures)
	Yrowi := make([]float64, nOutputs, nOutputs)
	for i := nSamples - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		copy(Xrowi, X.RawRowView(i))
		X.SetRow(i, X.RawRowView(j))
		X.SetRow(j, Xrowi)
		copy(Yrowi, Y.RawRowView(i))
		Y.SetRow(i, Y.RawRowView(j))
		Y.SetRow(j, Yrowi)
	}
}

// DenseSigmoid put emelent-wise sigmoid of X into dst
func DenseSigmoid(dst *mat.Dense, X mat.Matrix) *mat.Dense {
	if dst == nil {
		r, c := X.Dims()
		dst = mat.NewDense(r, c, nil)
	}
	dst.Apply(func(i int, j int, v float64) float64 {
		return 1. / (1. + math.Exp(-v))
	}, X)
	return dst
}

func unused(...interface{}) {}
