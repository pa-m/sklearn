package linearModel

import (
	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// LogisticRegression WIP
type LogisticRegression struct {
	RegularizedRegression
}

// NewLogisticRegression create and init a *LogisticRegression
func NewLogisticRegression() *LogisticRegression {
	regr := &LogisticRegression{}
	//regr.StepSize = 1e-4
	regr.Tol = 1e-6
	//regr.Optimizer = base.NewAdamOptimizer()
	regr.FitIntercept = true
	regr.Normalize = false
	regr.ActivationFunction = base.Logistic{}
	regr.LossFunction = CrossEntropyLoss
	regr.Options.GOMethodCreator = func() optimize.Method { return &optimize.LBFGS{} }
	return regr
}

// PredictProba predicts probabolity of y=1 for X using Coef
func (regr *LogisticRegression) PredictProba(X, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)
	Y.Apply(func(i int, o int, y float64) float64 {
		return (base.Sigmoid{}).F(y)
	}, Y)
}

// Predict predicts y for X using Coef
func (regr *LogisticRegression) Predict(X, Y *mat.Dense) {
	regr.PredictProba(X, Y)
	nSamples, nOutputs := Y.Dims()
	for i := 0; i < nSamples; i++ {
		o1 := floats.MaxIdx(Y.RawRowView(i))
		for o := 0; o < nOutputs; o++ {
			v := 0.
			if o == o1 {
				v = 1.
			}
			Y.Set(i, o, v)
		}
	}
}
