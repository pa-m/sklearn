package base

import (
	"gonum.org/v1/gonum/mat"
)

// Fiter have a Fit(Matrix,Matrix) Fitter method. may be a Predicter or a Transformer
type Fiter interface {
	Fit(X, Y mat.Matrix) Fiter
}

// Predicter have Predict(Matrix,Mutable). if 2nd arg (result receiver) is nil, il will be allocated and returned by Predict
type Predicter interface {
	Fiter
	GetNOutputs() int
	Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense
	Score(X, Y mat.Matrix) float64
	IsClassifier() bool
	PredicterClone() Predicter
}

// Transformer transforms X,Y into Xout,Yout
type Transformer interface {
	Fiter
	Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense)
	FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense)
	TransformerClone() Transformer
}
