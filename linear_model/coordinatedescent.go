package linearModel

import (
	"strings"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
)

// ElasticNetRegression is the struct for coordinate descent regularized regressions: ElasticNet,Ridge,Lasso
// Selection is cyclic or random. defaults to cyclic
type ElasticNetRegression struct {
	LinearRegression
	Tol, Alpha, L1Ratio float64
	MaxIter             int
	Selection           string
	WarmStart, Positive bool
	CDResult            *CDResult
}

// Fit ElasticNetRegression with coordinate descent
func (regr *ElasticNetRegression) Fit(X0, Y0 *mat.Dense) base.Transformer {
	var X, Y, YOffset *mat.Dense
	X, Y, regr.XOffset, YOffset, regr.XScale = PreprocessData(X0, Y0, regr.FitIntercept, regr.Normalize, nil)
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()

	l1reg := regr.Alpha * regr.L1Ratio * float64(NSamples)
	l2reg := regr.Alpha * (1. - regr.L1Ratio) * float64(NSamples)
	if !regr.WarmStart {
		regr.Coef = mat.NewDense(NFeatures, NOutputs, nil)
	}
	random := strings.EqualFold("random", regr.Selection)
	if NOutputs == 1 {
		y := &mat.VecDense{}
		w := &mat.VecDense{}

		w.ColViewOf(regr.Coef, 0)
		y.ColViewOf(Y, 0)
		regr.CDResult = enetCoordinateDescent(w, l1reg, l2reg, X, y, regr.MaxIter, regr.Tol, nil, random, regr.Positive)

	} else {
		regr.CDResult = enetCoordinateDescentMultiTask(regr.Coef, l1reg, l2reg, X, Y, regr.MaxIter, regr.Tol, nil, random, regr.Positive)
	}
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// NewElasticNet creates a *RegularizedRegression with Alpha=1 and L1Ratio=0.5
func NewElasticNet() *ElasticNetRegression {
	return NewMultiTaskElasticNet()
}

// NewMultiTaskElasticNet creates a *ElasticNetRegression with Alpha=1 and L1Ratio=0.5
func NewMultiTaskElasticNet() *ElasticNetRegression {
	regr := &ElasticNetRegression{}
	regr.FitIntercept = true
	regr.Tol = 1e-4
	regr.Alpha = 1.
	regr.L1Ratio = .5
	regr.MaxIter = 1000
	regr.Selection = "cyclic"
	return regr
}

//NewLasso creates a *ElasticNetRegression with Alpha=1 and L1Ratio = 1
func NewLasso() *ElasticNetRegression {
	m := NewMultiTaskElasticNet()
	m.L1Ratio = 1.
	return m
}

// NewMultiTaskLasso creates a *RegularizedRegression with Alpha=1 and L1Ratio=1
func NewMultiTaskLasso() *ElasticNetRegression {
	m := NewMultiTaskElasticNet()
	m.L1Ratio = 1.
	return m
}
