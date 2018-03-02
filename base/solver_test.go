package base

import (
	"fmt"

	lm "github.com/pa-m/sklearn/linear_model"
	//"github.com/pa-m/sklearn/metrics"
	"math/rand"
	"testing"
	"time"

	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

type Problem struct {
	X, Theta, Y   *mat.Dense
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
	return &Problem{X: X, Theta: TrueTheta, Y: Ytrue}
}

func NewRandomSigmoidProblem(nSamples, nFeatures, nOutputs int) *Problem {
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	DenseSigmoid(p.Y, p.Y)
	return p
}

func (p *Problem) Normalize() *Problem {
	preprocessing.DenseNormalize(p.X, true, true)
	preprocessing.DenseNormalize(p.Y, true, true)
	return p
}

type Regression interface {
	Fit(X, Y *mat.Dense)
	Predict(X, Y *mat.Dense)
	Score(X, Y *mat.Dense)
	Coef() *mat.Dense
	Intercept() *mat.Dense
}

func testSolver(t *testing.T, name string, s Optimizer, p *Problem) {
	X, _, Ytrue := p.X, p.Theta, p.Y
	start := time.Now()
	res := lm.LinFit(X, Ytrue, &lm.LinFitOptions{Epochs: 1000, MiniBatchSize: p.MiniBatchSize, Tol: 1e-3, Solver: s, Alpha: 0})
	if !res.Converged && res.RMSE > 1e-3 {
		t.Errorf("%s RMSE:%g", name, res.RMSE)
	}
	fmt.Printf("%s ok. RMSE=%.9g epochs=%d elapsed=%s\n", name, res.RMSE, res.Epoch, time.Since(start))

	//fmt.Println("Theta true", TrueTheta)
	//fmt.Println("Theta Pred", s.Theta)
	//fmt.Println("Test ", name, time.Since(start))
}
