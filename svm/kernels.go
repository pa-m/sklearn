package svm

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// Kernel is the interface for kernels
type Kernel interface {
	Func(a, b []float64) float64
}

// LinearKernel is dot product
type LinearKernel struct{}

// Func for LinearKernel
func (LinearKernel) Func(a, b []float64) (sumprod float64) {
	sumprod = floats.Dot(a, b)
	// for i := range a {
	// 	sumprod += a[i] * b[i]
	// }
	return
}

// PolynomialKernel ...
type PolynomialKernel struct{ gamma, coef0, degree float64 }

// Func for PolynomialKernel
func (kdata PolynomialKernel) Func(a, b []float64) (sumprod float64) {
	return math.Pow(kdata.gamma*floats.Dot(a, b)+kdata.coef0, kdata.degree)
}

// RBFKernel ...
type RBFKernel struct{ gamma float64 }

// Func for RBFKernel
func (kdata RBFKernel) Func(a, b []float64) float64 {
	L2 := 0.
	for i := range a {
		v := a[i] - b[i]
		L2 += v * v
	}
	return math.Exp(-kdata.gamma * L2)
}

// SigmoidKernel ...
type SigmoidKernel struct{ gamma, coef0 float64 }

// Func for SigmoidKernel
func (kdata SigmoidKernel) Func(a, b []float64) (sumprod float64) {
	return math.Tanh(kdata.gamma*floats.Dot(a, b) + kdata.coef0)
}
