package linearModel

import (
	"math"
)

// Activation ...
type Activation interface {
	F(x float64) float64
	Fprime(x, y float64) float64
}

// see https://en.wikipedia.org/wiki/Activation_function

// Identity ...
type Identity struct{}

// F ...
func (Identity) F(x float64) float64 { return x }

// Fprime ...
func (Identity) Fprime(x, y float64) float64 { return 1 }

// Logistic ...
type Logistic struct{}

// F ...
func (Logistic) F(x float64) float64 { return 1./1 + math.Exp(-x) }

// Fprime ...
func (Logistic) Fprime(x, y float64) float64 { return y * (1 - y) }

// Sigmoid ...
type Sigmoid = Logistic

// Tanh ...
type Tanh struct{}

// F ...
func (Tanh) F(x float64) float64 { return math.Tanh(x) }

// Fprime ...
func (Tanh) Fprime(x, y float64) float64 { return 1. - y*y }

// ReLU ...
type ReLU struct{}

// F ...
func (ReLU) F(x float64) float64 {
	if x < 0. {
		return 0.
	}
	return x
}

// Fprime ...
func (ReLU) Fprime(x, y float64) float64 {
	if x < 0. {
		return 0.
	}
	return 1.
}
