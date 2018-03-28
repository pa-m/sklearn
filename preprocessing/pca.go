package preprocessing

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// PCA is a thin single value decomposition transformer
type PCA struct {
	mat.SVD
	SingularValues, ExplainedVarianceRatio []float64
}

// NewPCA returns a *PCA
func NewPCA() *PCA { return &PCA{} }

// Fit computes the svd of X
func (m *PCA) Fit(X, Y *mat.Dense) Transformer {
	_, c := X.Dims()
	m.SVD.Factorize(X, mat.SVDThin)
	m.SingularValues = make([]float64, c)
	m.ExplainedVarianceRatio = make([]float64, c)
	m.SVD.Values(m.SingularValues)
	floats.MulTo(m.ExplainedVarianceRatio, m.SingularValues, m.SingularValues)
	floats.Scale(1./floats.Sum(m.ExplainedVarianceRatio), m.ExplainedVarianceRatio)
	return m
}

// Transform Transforms X
func (m *PCA) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	var v = new(mat.Dense)
	m.SVD.VTo(v)
	nSamples, _ := X.Dims()
	_, vCols := v.Dims()
	Xout = mat.NewDense(nSamples, vCols, nil)
	Xout.Mul(X, v)
	Yout = Y
	return
}

// FitTransform for PCA
func (m *PCA) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return m.Fit(X, Y).Transform(X, Y)
}

// InverseTransform put X into original space
func (m *PCA) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	if X == nil {
		return X, Y
	}

	var v = new(mat.Dense)
	m.SVD.VTo(v)
	nSamples, _ := X.Dims()
	_, vCols := v.Dims()
	Xout = mat.NewDense(nSamples, vCols, nil)
	Xout.Mul(X, v.T())
	Yout = Y
	return
}
