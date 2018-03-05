package neuralNetwork

import (
	"math/rand"
	"testing"

	"github.com/pa-m/sklearn/base"
	lm "github.com/pa-m/sklearn/linear_model"

	"gonum.org/v1/gonum/mat"
)

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func NewRandomLinearProblem(nSamples, nFeatures, nOutputs int) *Problem {
	X := mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return 1.
		}
		return rand.NormFloat64() * 10.
	}, X)
	TrueTheta := mat.NewDense(nFeatures, nOutputs, nil)
	TrueTheta.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64() * 10.
	}, TrueTheta)
	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
	Ytrue.Product(X, TrueTheta)
	return &Problem{X: X, Y: Ytrue}
}

func TestMLPRegressor(t *testing.T) {
	p := NewRandomLinearProblem(1000, 2, 2)
	regr := NewMLPRegressor([]int{2, 2}, lm.Logistic{}, base.NewAdamOptimizer())

	regr.Fit(p.X, p.Y)

}
