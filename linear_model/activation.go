package linearModel

import (
	"gonum.org/v1/gonum/mat"
	//"gonum.org/v1/gonum/mat"
	"math"
)

// Activation is the inteface for an activation function
type Activation interface {
	F(x float64) float64
	Fprime(y float64) float64
}

// Activations is the map of implemented activation functions
var Activations = map[string]Activation{"identity": Identity{}, "logistic": Logistic{}, "relu": ReLU{}, "tanh": Tanh{}}

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

// Softmax ...
type Softmax struct{}

// F for Softmax
func (Softmax) F(z mat.Vector, output int) float64 {
	outputs, _ := z.Dims()
	var N, D float64
	for o := 0; o < outputs; o++ {
		expzo := math.Exp(z.AtVec(o))
		D += expzo
		if o == output {
			N += expzo
		}
	}
	return N / D
}

// Fprime ... DjSi = Si (1(i=j)-Sj)
