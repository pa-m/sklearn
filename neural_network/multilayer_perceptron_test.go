package neuralNetwork

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
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
		return activation.F(xtheta)
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
	testMLPRegressor(t, lm.Identity{}, "square", func() base.Optimizer { return base.NewAdamOptimizer() }, 4)
}

func TestMLPRegressorLogisticLogLoss(t *testing.T) {
	testMLPRegressor(t, lm.Logistic{}, "log", func() base.Optimizer { return base.NewAdamOptimizer() }, 2)
}

func TestMLPRegressorLogisticCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, lm.Logistic{}, "cross-entropy", func() base.Optimizer { return base.NewAdamOptimizer() }, 2)
}

// Does it make sense ???
// func TestMLPRegressorTanhCrossEntropyLoss(t *testing.T) {
// 	testMLPRegressor(t, lm.Tanh{}, "cross-entropy", func() base.Optimizer { return base.NewAdamOptimizer() }, 2)
// }

// Does it make sense ???
// func TestMLPRegressorReLUCrossEntropyLoss(t *testing.T) {
// 	testMLPRegressor(t, lm.ReLU{}, "cross-entropy", func() base.Optimizer { return base.NewAdamOptimizer() }, 2)
// }

func testMLPRegressor(t *testing.T, activation lm.Activation, lossName string, OptimCreator func() base.Optimizer, maxLayers int) {
	var nSamples, nFeatures, nOutputs = 2000, 3, 2
	var p = NewRandomProblem(nSamples, nFeatures, nOutputs, activation, lossName)
	var HiddenLayerSizes []int
	for l := 0; l < maxLayers; l++ {
		regr := NewMLPRegressor(HiddenLayerSizes, activation, nil)
		regr.Loss = lm.LossFunctions[lossName]
		regr.SetOptimizer(OptimCreator, true)
		regr.Epochs = 200
		testSetup := fmt.Sprintf("%T %T %s layers %v", regr, activation, lossName, HiddenLayerSizes)
		//Ypred := mat.NewDense(nSamples, nOutputs, nil)
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
