package linearModel

import (
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

func (regr *RegularizedRegression) fitcd(X0, Y0 *mat.Dense) base.Transformer {
	X := mat.DenseCopyOf(X0)
	regr.XOffset, regr.XScale = preprocessing.DenseNormalize(X, regr.FitIntercept, regr.Normalize)
	Y := mat.DenseCopyOf(Y0)
	YOffset, _ := preprocessing.DenseNormalize(Y, regr.FitIntercept, false)
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()

	l1reg := regr.Alpha * regr.L1Ratio * float64(NSamples)
	l2reg := regr.Alpha * (1. - regr.L1Ratio) * float64(NSamples)
	regr.Coef = mat.NewDense(NFeatures, NOutputs, nil)
	if NOutputs <= 1 {
		for o := 0; o < NOutputs; o++ {
			y := &mat.VecDense{}
			w := &mat.VecDense{}
			w.ColViewOf(regr.Coef, o)
			y.ColViewOf(Y, o)
			enetCoordinateDescent(w, l1reg, l2reg, X, y, regr.MaxIter, regr.Tol, nil, regr.Random, regr.Positive)
		}
	} else {
		enetCoordinateDescentMultiTask(regr.Coef, l1reg, l2reg, X, Y, regr.MaxIter, regr.Tol, nil, regr.Random, regr.Positive)
	}
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

//NewLasso creates a *RegularizedRegression with Alpha=1 and L1Ratio = 1
func NewLasso() *RegularizedRegression {
	regr := &RegularizedRegression{}
	regr.FitIntercept = true
	regr.Tol = 0.0001
	regr.LossFunction = SquareLoss
	regr.Alpha = 1.
	regr.L1Ratio = 1.
	regr.MaxIter = 1000
	return regr
}

// NewElasticNet creates a *RegularizedRegression with Alpha=1 and L1Ratio=0.5
func NewElasticNet() *RegularizedRegression {
	regr := &RegularizedRegression{}
	regr.FitIntercept = true
	regr.Tol = 1e-6
	regr.LossFunction = SquareLoss
	regr.Alpha = 1.
	regr.L1Ratio = .5
	return regr
}

// NewMultiTaskElasticNet creates a *RegularizedRegression with Alpha=1 and L1Ratio=0.5
func NewMultiTaskElasticNet() *RegularizedRegression {
	regr := &RegularizedRegression{}
	regr.FitIntercept = true
	regr.Tol = 1e-4
	regr.LossFunction = SquareLoss
	regr.Alpha = 1.
	regr.L1Ratio = .5
	regr.MaxIter = 1000
	return regr
}

// NewMultiTaskLasso creates a *RegularizedRegression with Alpha=1 and L1Ratio=1
func NewMultiTaskLasso() *RegularizedRegression {
	regr := &RegularizedRegression{}
	regr.FitIntercept = true
	regr.Tol = 1e-4
	regr.LossFunction = SquareLoss
	regr.Alpha = 1.
	regr.L1Ratio = 1.
	regr.MaxIter = 1000
	return regr
}
