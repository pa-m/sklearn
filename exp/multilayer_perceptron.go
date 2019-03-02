package exp

import (
	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

// MLPRegressor ...
type MLPRegressor struct{ BaseMultilayerPerceptron64 }

// Regressors is the list of regressors in this package
var Regressors = []base.Regressor{&MLPRegressor{}}

// NewMLPRegressor returns a *MLPRegressor with defaults
// activation is one of identity,logistic,tanh,relu
// solver is on of sgd,adam  defaults to "adam"
// Alpha is the regularization parameter
func NewMLPRegressor(hiddenLayerSizes []int, activation string, solver string, Alpha float64) *MLPRegressor {
	regr := &MLPRegressor{
		BaseMultilayerPerceptron64: *NewBaseMultilayerPerceptron64(),
	}
	regr.HiddenLayerSizes = hiddenLayerSizes
	regr.Activation = activation
	regr.Solver = solver
	regr.Alpha = Alpha
	return regr
}

// Clone ...
func (regr *MLPRegressor) Clone() base.Transformer {
	clone := *regr
	return &clone
}

// Fit ...
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) base.Transformer {
	regr.fit(X.RawMatrix(), Y.RawMatrix(), false)
	return regr
}

// Predict return the forward result
func (regr *MLPRegressor) Predict(X, Y *mat.Dense) base.Regressor {
	regr.BaseMultilayerPerceptron64.predict(X.RawMatrix(), Y.RawMatrix())
	return regr
}

// FitTransform is for Pipeline
func (regr *MLPRegressor) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	r, c := Y.Dims()
	Xout, Yout = X, mat.NewDense(r, c, nil)
	regr.Fit(X, Y)
	regr.Predict(X, Yout)
	return
}

// Transform is for Pipeline
func (regr *MLPRegressor) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	r, c := Y.Dims()
	Xout, Yout = X, mat.NewDense(r, c, nil)
	regr.Predict(X, Yout)
	return
}

// Score for MLPRegressor returns R2Score
func (regr *MLPRegressor) Score(X, Y *mat.Dense) float64 {
	nSamples, _ := X.Dims()
	nOutputs := regr.NOutputs
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	regr.Predict(X, Y)
	return r2Score64(Y.RawMatrix(), Ypred.RawMatrix())
}

// MLPClassifier ...
type MLPClassifier struct{ BaseMultilayerPerceptron64 }

// NewMLPClassifier returns a *MLPClassifier with defaults
// activation is one of logistic,tanh,relu
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// lossName is one of square,log,cross-entropy (one of the keys of lm.LossFunctions) defaults to "log"
func NewMLPClassifier(hiddenLayerSizes []int, activation string, solver string, Alpha float64) *MLPClassifier {
	regr := &MLPClassifier{
		BaseMultilayerPerceptron64: *NewBaseMultilayerPerceptron64(),
	}
	regr.HiddenLayerSizes = hiddenLayerSizes
	regr.Activation = activation
	regr.Solver = solver
	regr.Alpha = Alpha
	return regr
}

// Fit ...
func (regr *MLPClassifier) Fit(X, Y *mat.Dense) base.Transformer {
	regr.BaseMultilayerPerceptron64.Fit(X, Y)
	return regr
}

// Predict return the forward result for MLPClassifier
func (regr *MLPClassifier) Predict(X, Y *mat.Dense) base.Regressor {
	nSamples, _ := X.Dims()
	nOutputs := regr.NOutputs
	yr, _ := Y.Dims()
	if yr == 0 {
		*Y = *mat.NewDense(nSamples, nOutputs, nil)
	}
	regr.BaseMultilayerPerceptron64.Predict(X, Y)
	return regr
}

// Transform for pipeline
func (regr *MLPClassifier) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	nOutputs := regr.NOutputs
	Yout = mat.NewDense(nSamples, nOutputs, nil)
	regr.Predict(X, Yout)
	Xout = X
	return
}

// Score for MLPClassifier computes accuracy score
func (regr *MLPClassifier) Score(X, Y *mat.Dense) float64 {
	Ypred := &mat.Dense{}
	regr.Predict(X, Ypred)

	return accuracyScore64(Y.RawMatrix(), Ypred.RawMatrix())
}
