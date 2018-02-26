package base

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
	"time"
)

type Problem struct {
	X, Theta, Y *mat.Dense
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

func TestAdamSolver(t *testing.T) {
	p := NewRandomLinearProblem(10000, 12, 7)
	s := NewAdamSolver()
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = .5
	testSolver(t, "AdamSolver", s, p)
}

func testSolver(t *testing.T, name string, s Optimizer, p *Problem) {
	X, _, Ytrue := p.X, p.Theta, p.Y
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()

	Theta := mat.NewDense(nFeatures, nOutputs, nil)
	Theta.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()
	}, Theta)

	epochs, miniBatchStart, miniBatchSize := 10000, 0, 200

	YpredMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	ErrMini := mat.NewDense(miniBatchSize, nOutputs, nil)

	grad := mat.NewDense(nFeatures, nOutputs, nil)

	Tol := 1e-3

	start := time.Now()
	s.SetTheta(Theta)
	var timeStep uint64
	mse := math.Inf(1)
	converged := false
	for epoch := 1; epoch <= epochs && !converged; epoch++ {
		DenseShuffle(X, Ytrue)
		seCumul := 0.
		for miniBatch := 0; miniBatch*miniBatchSize < nSamples; miniBatch++ {
			miniBatchStart = miniBatch * miniBatchSize
			Xmini := (X.Slice(miniBatchStart, miniBatchStart+miniBatchSize, 0, nFeatures))
			YtrueMini := (Ytrue.Slice(miniBatchStart, miniBatchStart+miniBatchSize, 0, nOutputs))
			YpredMini.Product(Xmini, Theta)
			ErrMini.Sub(YpredMini, YtrueMini)

			// compute the gradient of squared error = 2*Xt*err
			grad.Product(Xmini.T(), ErrMini)
			grad.Scale(2., grad)

			s.UpdateParams(grad)
			ErrMini.Apply(func(i, j int, v float64) float64 {
				seCumul += v * v
				return v
			}, ErrMini)
		}
		mse = seCumul / float(nSamples)
		converged = mse < Tol*Tol
		timeStep = s.GetTimeStep()

		if timeStep%20 == 0 || converged {
			fmt.Println(timeStep, "RMSE:", math.Sqrt(mse))
		}

	}
	//fmt.Println("Theta true", TrueTheta)
	//fmt.Println("Theta Pred", s.Theta)
	fmt.Println("Test ", name, time.Since(start))
}
