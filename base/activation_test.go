package base

import (
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
)

func TestIdentity(t *testing.T) {
	testActivationDerivatives(t, Identity{})
}

func TestLogistic(t *testing.T) {
	testActivationDerivatives(t, Logistic{})
}

func TestTanh(t *testing.T) {
	testActivationDerivatives(t, Tanh{})
}
func TestReLU(t *testing.T) {
	testActivationDerivatives(t, ReLU{})
}

func testActivationDerivatives(t *testing.T, activation Activation) {
	for pass := 0; pass < 5; pass++ {
		x := rand.NormFloat64()
		theta := rand.NormFloat64()
		y := activation.F(x * theta)
		var expected = fd.Derivative(activation.F, x*theta, &fd.Settings{Step: 1e-6})

		actual := activation.Fprime(y)

		if !floats.EqualWithinAbs(expected, actual, 1e-3) {
			t.Errorf("testActivationDerivatives %T x:%g theta:%g y:%g expected:%g actual:%g", activation, x, theta, y, expected, actual)
		}
	}
}
