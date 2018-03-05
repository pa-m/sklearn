package neuralNetwork

import (
	"math/rand"
	"testing"

	lm "github.com/pa-m/sklearn/linear_model"

	"gonum.org/v1/gonum/mat"
)

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func NewRandomProblem(nSamples, nFeatures, nOutputs int) *Problem {
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
		return lm.Logistic{}.F(xtheta)
	}, Ytrue)
	return &Problem{X: X, Y: Ytrue}
}

func TestMLPRegressor(t *testing.T) {
	nSamples, nFeatures, nOutputs := 1000, 3, 2
	p := NewRandomProblem(nSamples, nFeatures, nOutputs)
	regr := NewMLPRegressor([]int{2}, lm.Logistic{}, nil)
	//Ypred := mat.NewDense(nSamples, nOutputs, nil)
	regr.Fit(p.X, p.Y)
	if regr.J > 1e-3 {
		t.Fail()
	}
}
