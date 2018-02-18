package base

import (
	"github.com/pa-m/sklearn/metrics"
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
