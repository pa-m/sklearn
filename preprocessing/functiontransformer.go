package preprocessing

import (
	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
)

// FunctionTransformer Constructs a transformer from an arbitrary callable.
type FunctionTransformer struct {
	Func, InverseFunc func(X, Y *mat.Dense) (X1, Y1 *mat.Dense)
}

// NewFunctionTransformer ...
func NewFunctionTransformer(f, invf func(X, Y *mat.Dense) (X1, Y1 *mat.Dense)) *FunctionTransformer {
	return &FunctionTransformer{
		Func:        f,
		InverseFunc: invf,
	}
}

// TransformerClone ...
func (m *FunctionTransformer) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit ...
func (m *FunctionTransformer) Fit(X, Y mat.Matrix) base.Fiter {
	return m
}

// Transform ...
func (m *FunctionTransformer) Transform(X, Y mat.Matrix) (X1, Y1 *mat.Dense) {
	X1, Y1 = m.Func(base.ToDense(X), base.ToDense(Y))
	return
}

// FitTransform fit to dat, then transform it
func (m *FunctionTransformer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform ...
func (m *FunctionTransformer) InverseTransform(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
	X1, Y1 = m.InverseFunc(X, Y)
	return
}
