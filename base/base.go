package base

import (
	"github.com/pa-m/sklearn/metrics"
	"math/rand"
)

type float = float64

// Predicter is an interface for Predict method
type Predicter interface {
	Predict([][]float) []float
}

// Predicter2 is an interface for Predict method
type Predicter2 interface {
	Predict([][]float) [][]float
}

// RegressorMixin is a base for predicters. provides a Score(X,w,weights) method
type RegressorMixin struct{ Predicter }

// Score returns R2Score of predicter
func (predicter *RegressorMixin) Score(X [][]float, y, sampleWeight []float) float {
	yPred := predicter.Predict(X)
	return metrics.R2Score(y, yPred, sampleWeight, "variance_weighted")
}

// RegressorMixin2 is a base for predicters. provides a Score(X,w,weights) method
type RegressorMixin2 struct{ Predicter2 }

// Score returns R2Score of predicter
func (predicter *RegressorMixin2) Score(X, y [][]float, sampleWeight []float) float {
	yPred := predicter.Predict(X)
	return metrics.R2Score2(y, yPred, sampleWeight, "variance_weighted")
}

// Shuffle shuffles X,y samples
func Shuffle(X [][]float, y []float) {
	for i := len(X) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		X[i], X[j] = X[j], X[i]
		y[i], y[j] = y[j], y[i]
	}
}
