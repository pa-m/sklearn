package linearModel

import (
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// LogisticRegression WIP
type LogisticRegression struct {
	RegularizedRegression
	OneHotEncoder *preprocessing.OneHotEncoder
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

// EncodeLabels applies a onehotencoder if Ytrue has only one column
func (regr *LogisticRegression) EncodeLabels(Ytrue *mat.Dense) *mat.Dense {
	Y := Ytrue
	_, nOutputs := Ytrue.Dims()
	if nOutputs == 1 {
		regr.OneHotEncoder = preprocessing.NewOneHotEncoder()
		_, Y = regr.OneHotEncoder.FitTransform(nil, Ytrue)
	}
	return Y
}

// Fit for LogisticRegression
func (regr *LogisticRegression) Fit(X, Ycls *mat.Dense) base.Transformer {
	Y := regr.EncodeLabels(Ycls)
	return regr.RegularizedRegression.Fit(X, Y)
}

// PredictProba predicts probability of y=1 for X using Coef
func (regr *LogisticRegression) PredictProba(X, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)
	Y.Apply(func(i int, o int, y float64) float64 {
		return regr.ActivationFunction.F(y)
	}, Y)
}

// Predict predicts y for X using Coef
func (regr *LogisticRegression) Predict(X, Ycls *mat.Dense) {
	var Y = Ycls
	if regr.OneHotEncoder != nil {
		Y = &mat.Dense{}
	}
	regr.PredictProba(X, Y)
	//nSamples, nOutputs := Y.Dims()
	if regr.OneHotEncoder != nil {
		_, Ycls1 := regr.OneHotEncoder.InverseTransform(nil, Y)
		if Ycls.IsZero() {
			Ycls.SetRawMatrix(Ycls1.RawMatrix())
		} else {
			Ycls.Copy(Ycls1)
		}

	}
}
