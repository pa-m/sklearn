package preprocessing

import "gonum.org/v1/gonum/mat"

// FunctionTransformer Constructs a transformer from an arbitrary callable.
type FunctionTransformer struct {
	Func, InverseFunc func(X, Y *mat.Dense) (X1, Y1 *mat.Dense)
}

// NewFunctionTransformer ...
func NewFunctionTransformer(f, invf func(X, Y *mat.Dense) (X1, Y1 *mat.Dense)) *FunctionTransformer {
	return &FunctionTransformer{Func: f, InverseFunc: invf}
}

// Fit ...
func (m *FunctionTransformer) Fit(X, Y *mat.Dense) Transformer {
	return m
}

// Transform ...
func (m *FunctionTransformer) Transform(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
	X1, Y1 = m.Func(X, Y)
	return
}

// InverseTransform ...
func (m *FunctionTransformer) InverseTransform(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
	X1, Y1 = m.InverseFunc(X, Y)
	return
}
