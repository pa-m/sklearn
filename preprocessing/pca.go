package preprocessing

import (
	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// PCA is a thin single value decomposition transformer
type PCA struct {
	mat.SVD
	MinVarianceRatio                       float64
	NComponents                            int
	SingularValues, ExplainedVarianceRatio []float64
}

// NewPCA returns a *PCA
func NewPCA() *PCA { return &PCA{} }

// TransformerClone ...
func (m *PCA) TransformerClone() base.Transformer {
	clone := *m
	return &clone
}

// Fit computes the svd of X
func (m *PCA) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	_, c := X.Dims()
	m.SVD.Factorize(X, mat.SVDThin)
	m.SingularValues = make([]float64, c)
	m.ExplainedVarianceRatio = make([]float64, c)
	m.SVD.Values(m.SingularValues)
	floats.MulTo(m.ExplainedVarianceRatio, m.SingularValues, m.SingularValues)
	floats.Scale(1./floats.Sum(m.ExplainedVarianceRatio), m.ExplainedVarianceRatio)

	if m.MinVarianceRatio > 0 {
		thres := m.MinVarianceRatio
		ExplainedVarianceRatio := 0.
		var nComponents int
		for nComponents = 0; nComponents < len(m.ExplainedVarianceRatio) && ExplainedVarianceRatio < thres; nComponents++ {
			ExplainedVarianceRatio += m.ExplainedVarianceRatio[nComponents]
		}
		m.NComponents = nComponents
	} else {
		if m.NComponents == 0 {
			m.NComponents = c
		}
	}

	return m
}

// Transform Transforms X
func (m *PCA) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	var v = new(mat.Dense)
	m.SVD.VTo(v)
	nSamples, _ := X.Dims()
	vRows, _ := v.Dims()
	Xout = mat.NewDense(nSamples, m.NComponents, nil)
	Xout.Mul(X, base.MatDenseSlice(v, 0, vRows, 0, m.NComponents))

	Yout = base.ToDense(Y)
	return
}

// FitTransform fit to dat, then transform it
func (m *PCA) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
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
