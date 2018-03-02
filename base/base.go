package base

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
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

// MatStr return a string from a mat.Matrix
func MatStr(Xs ...mat.Matrix) string {
	if len(Xs) == 0 {
		return ""
	}
	nSamples, nFeatures := Xs[0].Dims()
	b := bytes.NewBuffer(nil)

	for i := 0; i < nSamples; i++ {
		for imat, X := range Xs {
			_, nFeatures = X.Dims()
			for j := 0; j < nFeatures; j++ {
				io.WriteString(b, fmt.Sprintf("%g", X.At(i, j)))
				if j < nFeatures-1 || imat < len(Xs)-1 {
					io.WriteString(b, "\t")
				} else {
					io.WriteString(b, "\n")
				}
			}
		}
	}
	return b.String()
}

// MatColStr return the string for a matrix column
func MatColStr(X mat.Matrix, j int) string {
	nSamples, _ := X.Dims()
	var t = make([]float64, nSamples)
	mat.Col(t, j, X)
	return fmt.Sprint(t)
}

// MatRowStr returns the string for a matrix row
func MatRowStr(X mat.Matrix, i int) string {
	_, nFeatures := X.Dims()
	var t = make([]float64, nFeatures)
	mat.Row(t, i, X)
	return fmt.Sprint(t)
}
