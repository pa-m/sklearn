package base

import (
	_ "fmt"
	"math"
)

type GD struct {
	RegressorMixin
	Epochs                             int
	LearningRate, Decay, Tol, Momentum float
	Coefs_                             []float
}

func NewGD() *GD {
	self := &GD{Epochs: 3000, LearningRate: 1e-3, Decay: .99, Tol: 1e-3, Momentum: .5}
	self.Predicter = self
	return self
}

// adapted from gdSolver from  https://github.com/ohheydom/linearregression/blob/master/linear_regression.go
// Gradient Descent algorithm.

func (gd *GD) Fit(x [][]float64, y []float64) *GD {
	nIter := gd.Epochs
	n, nFeatures := len(x), len(x[0])
	gamma := gd.LearningRate / float(n)
	w := make([]float64, nFeatures+1)
	dw := make([]float64, nFeatures+1)
	gd.Coefs_ = w
	errors := make([]float64, n)
	for i := 0; i < nIter; i++ {
		predY := gd.Predict(x)
		errorSum := 0.0
		errorSum2 := 0.0
		// compute errorSum for updating w[0] and errorSum2 for Tol check
		for j := 0; j < n; j++ {
			errors[j] = y[j] - predY[j]
			errorSum += errors[j]
			errorSum2 += math.Pow(errors[j], 2.)

		}
		// update w[1]..w[n]
		for k := 0; k < n; k++ {
			for l := 1; l < len(w); l++ {
				dw[l] = gd.Momentum*dw[l] + gamma*x[k][l-1]*errors[k]
				w[l] += dw[l]
			}
		}
		// update w[0]
		w[0] += gamma * errorSum
		//fmt.Printf("GD %v %v\n", w, errorSum)

		//decrease lr/n
		gamma *= gd.Decay
		//check Tol
		if math.Sqrt(errorSum2)/float(n) < gd.Tol {
			break
		}
	}
	gd.Coefs_ = w
	return gd
}

// predY uses the given weights to calculate each sample's label.
func (gd *GD) Predict(x [][]float64) []float64 {
	w := gd.Coefs_
	n, nFeatures := len(x), len(x[0])
	predY := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 1; j <= nFeatures; j++ {
			predY[i] += x[i][j-1] * w[j]
		}
		predY[i] += w[0]
	}
	return predY
}
