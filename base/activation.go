package base

import (
	"math"
)

// Activations is the map of implemented activation functions
var Activations map[string]Activation

func init() { // go tip don't want this initialization wthout init
	Activations = map[string]Activation{"identity": Identity{}, "logistic": Logistic{}, "relu": ReLU{}, "tanh": Tanh{}}
}

// see https://en.wikipedia.org/wiki/Activation_function

// Identity ...
type Identity struct{}

// F ...
func (Identity) F(x float64) float64 { return x }

// Fprime ...y=F(X*Theta)
func (Identity) Fprime(y float64) float64 { return 1 }

// Logistic ...
type Logistic struct{}

// F ...
func (Logistic) F(x float64) float64 { return 1. / (1. + math.Exp(-x)) }

// Fprime ...
func (Logistic) Fprime(y float64) float64 { return y * (1 - y) }

// Sigmoid ...
type Sigmoid = Logistic

// Tanh ...
type Tanh struct{}

// F ...
func (Tanh) F(x float64) float64 { return math.Tanh(x) }

// Fprime ...
func (Tanh) Fprime(y float64) float64 { return 1. - y*y }

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
func (ReLU) Fprime(y float64) float64 {
	if y <= 0. {
		return 0.
	}
	return 1.
}

// Fprime ... DjSi = Si (1(i=j)-Sj)

// Activation is the inteface for an activation function
type Activation interface {
	F(x float64) float64
	Fprime(y float64) float64
}
