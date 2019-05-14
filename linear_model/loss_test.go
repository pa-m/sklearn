package linearmodel

import (
	"fmt"
	"math"
	"testing"

	"golang.org/x/exp/rand"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestSquareLoss(t *testing.T) {
	for _, activationFunction := range base.Activations {
		testLossDerivatives(t, SquareLoss, activationFunction)
	}
}

func TestLogLoss(t *testing.T) {
	for _, activationFunction := range base.Activations {
		testLossDerivatives(t, LogLoss, activationFunction)
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	for _, activationFunction := range base.Activations {
		testLossDerivatives(t, CrossEntropyLoss, activationFunction)
	}
}

func testLossDerivatives(t *testing.T, lossFunc Loss, activation Activation) {
	nSamples := 100
	unused(fd.Gradient, floats.Sum, fmt.Println)

	Ypred := mat.NewDense(nSamples, 1, nil)
	Ydiff := mat.NewDense(nSamples, 1, nil)
	grad := mat.NewDense(1, 1, nil)
	Theta := mat.NewDense(1, 1, nil)
	{
		p := NewRandomLinearProblem(nSamples, 1, 1)
		// scaler := preprocessing.NewMinMaxScaler([]float64{0, 1})
		// X, Ytrue := scaler.Fit(p.X, p.Y).Transform(p.X, p.Y)
		X, Ytrue := p.X, p.Y
		Ytrue.Apply(func(i int, j int, _ float64) float64 {
			v := rand.Float64()
			return activation.F(v)
		}, Ytrue)
		//fmt.Printf("TestLoss %#v\n", activation)
		for i := 0; i < 10; i++ {
			Alpha := 0.
			L1Ratio := 0.

			theta0 := 2e-3 + .996*rand.Float64()
			eps := 1e-6
			Theta.Set(0, 0, theta0+eps)
			J1 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation, true)
			Theta.Set(0, 0, theta0-eps)

			J0 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation, true)
			Theta.Set(0, 0, theta0)
			lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation, true)

			dJ := (J1 - J0) / 2. / eps
			//fmt.Printf("%t %g %g\n", activation, grad.At(0, 0), dJ)
			if math.Abs(dJ-grad.At(0, 0)) > 1e-2 {
				fmt.Printf("%T %g %g\n", activation, grad.At(0, 0), dJ)
				t.Fail()
			}
		}
	}
}

func TestLossRegularization(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	Theta := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	XTheta := &mat.Dense{}
	XTheta.Mul(X, Theta)
	YPred := &mat.Dense{}
	Ydiff := &mat.Dense{}
	grad := &mat.Dense{}
	J := SquareLoss(XTheta, X, Theta, YPred, Ydiff, grad, 1, 0, 2, base.Identity{}, false)
	e := 7.5
	if math.Abs(e-J) > 1e-6 {
		t.Errorf("Expected:%g got %g", e, J)
	}
	J = SquareLoss(XTheta, X, Theta, YPred, Ydiff, grad, 1, 1, 2, base.Identity{}, false)
	e = (math.Sqrt(5) + math.Sqrt(25)) / 4
	if math.Abs(e-J) > 1e-6 {
		t.Errorf("Expected:%g got %g", e, J)
	}

}
