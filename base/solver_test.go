package base

import (
	"fmt"
	lm "github.com/pa-m/sklearn/linear_model"
	//"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
	"time"
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
	res := lm.LinFit(X, Ytrue, &lm.LinFitOptions{Epochs: 1000, MiniBatchSize: p.MiniBatchSize, Tol: 1e-3, Solver: s})
	if !res.Converged {
		t.Errorf("%s RMSE:%g", name, res.RMSE)
	}
	fmt.Printf("%s ok. RMSE=%.9g epochs=%d elapsed=%s\n", name, res.RMSE, res.Epoch, time.Since(start))

	//fmt.Println("Theta true", TrueTheta)
	//fmt.Println("Theta Pred", s.Theta)
	//fmt.Println("Test ", name, time.Since(start))
}

func testSigSolver(t *testing.T, name string, s Optimizer, p *Problem) {
	X, _, Ytrue := p.X, p.Theta, p.Y
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()

	Theta := mat.NewDense(nFeatures, nOutputs, nil)
	Theta.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()
	}, Theta)

	epochs, miniBatchStart, miniBatchSize := 1000, 0, 200
	if p.MiniBatchSize > 0 {
		miniBatchSize = p.MiniBatchSize
	}

	YpredMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	YdiffMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	ErrMini := mat.NewDense(miniBatchSize, nOutputs, nil)

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Err := mat.NewDense(nSamples, nOutputs, nil)

	grad := mat.NewDense(nFeatures, nOutputs, nil)

	Tol := 1e-3

	start := time.Now()
	s.SetTheta(Theta)
	var timeStep uint64
	ErrorSum := math.Inf(1)
	converged := false
	for epoch := 1; epoch <= epochs && !converged; epoch++ {
		DenseShuffle(X, Ytrue)
		ErrorSum = 0.
		for miniBatch := 0; miniBatch*miniBatchSize < nSamples; miniBatch++ {
			miniBatchStart = miniBatch * miniBatchSize
			Xmini := (X.Slice(miniBatchStart, miniBatchStart+miniBatchSize, 0, nFeatures))
			YtrueMini := (Ytrue.Slice(miniBatchStart, miniBatchStart+miniBatchSize, 0, nOutputs))
			YpredMini.Product(Xmini, Theta)
			DenseSigmoid(YpredMini, YpredMini)
			ErrMini.Apply(func(i int, j int, v float64) float64 {
				v = -YtrueMini.At(i, j)*math.Log(YpredMini.At(i, j)) - (1.-YtrueMini.At(i, j))*math.Log(1.-YpredMini.At(i, j))
				v /= float64(miniBatchSize)
				return v
			}, ErrMini)

			// compute the gradient
			YdiffMini.Sub(YpredMini, YtrueMini)
			grad.Product(Xmini.T(), YdiffMini)
			grad.Scale(1./float64(miniBatchSize), grad)

			s.UpdateParams(grad)
			MiniErrorSum := 0.
			ErrMini.Apply(func(i, j int, v float64) float64 {
				MiniErrorSum += v
				return v
			}, ErrMini)
			if epoch >= epochs-1 {
				fmt.Printf("%s mini rmse = %.6f\n", name, MiniErrorSum)
			}
			ErrorSum += MiniErrorSum
		}

		//d := func(X mat.Matrix) string { r, c := X.Dims(); return fmt.Sprintf("%d,%d", r, c) }
		Ypred.Mul(X, Theta)
		DenseSigmoid(Ypred, Ypred)
		ErrorSum = 0.
		Err.Apply(func(i int, j int, v float64) float64 {
			v = -Ytrue.At(i, j)*math.Log(Ypred.At(i, j)) - (1.-Ytrue.At(i, j))*math.Log(1.-Ypred.At(i, j))
			v /= float64(nSamples)
			ErrorSum += v
			return v
		}, Err)

		converged = ErrorSum < Tol*Tol
		timeStep = s.GetTimeStep()

	}
	unused(timeStep, fmt.Println, start)
	if !converged {
		t.Errorf("%s ErrorSum:%g", name, ErrorSum)
	}
	fmt.Printf("%s ok. ErrorSum=%.9g elapsed=%s\n", name, ErrorSum, time.Since(start))

	//fmt.Println("Theta true", TrueTheta)
	//fmt.Println("Theta Pred", s.Theta)
	//fmt.Println("Test ", name, time.Since(start))
}
