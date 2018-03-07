package neuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/gonum/floats"
	lm "github.com/pa-m/sklearn/linear_model"

	"gonum.org/v1/gonum/mat"
)

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func NewRandomProblem(nSamples, nFeatures, nOutputs int, activation Activation, loss string) *Problem {
	X := mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, X)
	TrueTheta := mat.NewDense(nFeatures, nOutputs, nil)
	TrueTheta.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, TrueTheta)
	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
	Ytrue.Product(X, TrueTheta)
	Ytrue.Apply(func(i, o int, xtheta float64) float64 {
		ytrue := activation.F(xtheta)
		if math.IsNaN(ytrue) {
			panic(fmt.Errorf("%T ytrue is NaN", activation))
		}
		return ytrue
	}, Ytrue)
	if loss == "cross-entropy" {
		for sample := 0; sample < nSamples; sample++ {
			oTrue := floats.MaxIdx(Ytrue.RawRowView(sample))
			for o := 0; o < nOutputs; o++ {
				y := 0.
				if o == oTrue {
					y = 1
				}
				Ytrue.Set(sample, o, y)
			}
		}
	}
	return &Problem{X: X, Y: Ytrue}
}

func TestMLPRegressorIdentitySquareLoss(t *testing.T) {
	testMLPRegressor(t, "identity", "square", "adam", 4)
}

func TestMLPRegressorLogisticLogLoss(t *testing.T) {
	testMLPRegressor(t, "logistic", "log", "adam", 2)
}

func TestMLPRegressorLogisticCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, "logistic", "cross-entropy", "adam", 2)
}

func TestMLPRegressorTanhCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, "tanh", "cross-entropy", "adam", 2)
}

func TestMLPRegressorReLUCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, "relu", "cross-entropy", "adam", 2)
}

func testMLPRegressor(t *testing.T, activationName string, lossName string, solver string, maxLayers int) {
	var nSamples, nFeatures, nOutputs = 2000, 3, 2
	activation := lm.Activations[activationName]
	var p = NewRandomProblem(nSamples, nFeatures, nOutputs, activation, lossName)
	var HiddenLayerSizes []int

	for l := 0; l < maxLayers; l++ {
		Alpha := 0.
		regr := NewMLPRegressor(HiddenLayerSizes, activationName, solver, Alpha)
		//regr.SetOptimizer(OptimCreator, true)
		regr.Epochs = 200
		testSetup := fmt.Sprintf("%T %T %s loss layers %v", regr, activation, lossName, HiddenLayerSizes)
		//Ypred := mat.NewDense(nSamples, nOutputs, nil)

		regr.Loss = lossName
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)

		if regr.J > 0.01 && regr.J > 0.5*regr.JFirst {
			t.Errorf("%s JFirst=%g J=%g", testSetup, regr.JFirst, regr.J)
		} else {
			fmt.Printf("%s ok J=%g elapsed=%s\n", testSetup, regr.J, elapsed)
		}
		HiddenLayerSizes = append(HiddenLayerSizes, 1+rand.Intn(9))
	}
}
