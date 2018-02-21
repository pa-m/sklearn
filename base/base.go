package base

import (
	"github.com/pa-m/sklearn/metrics"
	"math/rand"
)

type float = float64

type Predicter interface {
	Predict([][]float) []float
}
type RegressorMixin struct{ Predicter }

func (self *RegressorMixin) Score(X [][]float, y, sample_weight []float) float {
	y_pred := self.Predict(X)
	return metrics.R2Score(y, y_pred, sample_weight, "variance_weighted")
}

func Shuffle(X [][]float, y []float) {
	n_samples := len(X)
	for i := range X {
		j := i + rand.Intn(n_samples-i)
		X[i], X[j] = X[j], X[i]
		if y != nil {
			y[i], y[j] = y[j], y[i]
		}
	}
}
