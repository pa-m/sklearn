package linearmodel

import (
	"math"
	"strings"
	"sync"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
)

// CDResult is the coordinate descent specific part in the regression result
type CDResult struct {
	Gap, Eps float64
	NIter    int
}

// ElasticNet is the struct for coordinate descent regularized regressions: ElasticNet,Ridge,Lasso
// Selection is cyclic or random. defaults to cyclic
type ElasticNet struct {
	LinearRegression
	Tol, Alpha, L1Ratio float64
	MaxIter             int
	Selection           string
	WarmStart, Positive bool
	CDResult            CDResult
}

// Lasso is an alias for ElasticNet
type Lasso = ElasticNet

// MultiTaskElasticNet is an alias for ElasticNet
type MultiTaskElasticNet = Lasso

// MultiTaskLasso is an alias for ElasticNet/Lasso
type MultiTaskLasso = Lasso

// IsClassifier returns false for ElasticNet
func (*ElasticNet) IsClassifier() bool { return false }

// PredicterClone for ElasticNet
func (regr *ElasticNet) PredicterClone() base.Predicter {
	clone := *regr
	return &clone
}

// GetNOutputs returns output columns number for Y to pass to predict
func (regr *ElasticNet) GetNOutputs() int {
	_, nOutputs := regr.Coef.Dims()
	return nOutputs
}

// Fit ElasticNetRegression with coordinate descent
func (regr *ElasticNet) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X0, Y0 := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
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
		regr.CDResult = *enetCoordinateDescent(w, l1reg, l2reg, X, y, regr.MaxIter, regr.Tol, nil, random, regr.Positive)

	} else {
		regr.CDResult = *enetCoordinateDescentMultiTask(regr.Coef, l1reg, l2reg, X, Y, regr.MaxIter, regr.Tol, nil, random, regr.Positive)
	}
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// NewElasticNet creates a *ElasticNet with Alpha=1 and L1Ratio=0.5
func NewElasticNet() *ElasticNet {
	return NewMultiTaskElasticNet()
}

// NewMultiTaskElasticNet creates a *ElasticNet with Alpha=1 and L1Ratio=0.5
func NewMultiTaskElasticNet() *MultiTaskElasticNet {
	regr := &ElasticNet{}
	regr.FitIntercept = true
	regr.Tol = 1e-4
	regr.Alpha = 1.
	regr.L1Ratio = .5
	regr.MaxIter = 1000
	regr.Selection = "cyclic"
	return regr
}

//NewLasso creates a *ElasticNetRegression with Alpha=1 and L1Ratio = 1
func NewLasso() *Lasso {
	m := NewMultiTaskElasticNet()
	m.L1Ratio = 1.
	return m
}

// NewMultiTaskLasso creates a *RegularizedRegression with Alpha=1 and L1Ratio=1
func NewMultiTaskLasso() *MultiTaskLasso {
	m := NewMultiTaskElasticNet()
	m.L1Ratio = 1.
	return m
}

func alphaGrid(X, Y *mat.Dense, L1Ratio, eps float64, alphas []float64) {
	NAlphas := len(alphas)
	alphaMax := 0.
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()
	row := make([]float64, NFeatures+NOutputs)
	for sample := 0; sample < NSamples; sample++ {
		copy(row[0:NFeatures], X.RawRowView(sample))
		copy(row[NFeatures:NFeatures+NOutputs], Y.RawRowView(sample))
		norm := mat.Norm(mat.NewVecDense(len(row), row), 2)
		if norm > alphaMax {
			alphaMax = norm
		}

	}
	alphaMax *= float64(NSamples) * L1Ratio
	minp, maxp := math.Log10(alphaMax*eps), math.Log10(alphaMax)
	incp := (maxp - minp) / float64(NAlphas-1)
	for i, p := 0, minp; i < NAlphas; i, p = i+1, p+incp {
		alphas[i] = math.Pow(10, p)
	}

}

// EnetPath Compute elastic net path with coordinate descent
// no preprocessing is done here, you must have called PreprocessData before
func EnetPath(X, Y *mat.Dense, L1Ratio, eps float64, NAlphas int, Alphas *[]float64, verbose, positive bool) (alphas []float64, coefs []*mat.Dense, dualGaps []float64, nIters []int) {
	alphas = make([]float64, NAlphas)
	if Alphas == nil {
		alphaGrid(X, Y, L1Ratio, eps, alphas)
	} else {
		copy(alphas, *Alphas)
	}
	NAlphas = len(alphas)
	coefs = make([]*mat.Dense, NAlphas)
	dualGaps = make([]float64, NAlphas)
	nIters = make([]int, NAlphas)
	wg := new(sync.WaitGroup)
	for ialpha, alpha := range alphas {
		wg.Add(1)
		go func(ialpha int, alpha float) {
			m := NewElasticNet()
			m.FitIntercept = false
			m.Normalize = false
			m.L1Ratio = L1Ratio
			m.Tol = eps
			m.Positive = positive
			m.Alpha = alpha
			m.Fit(X, Y)
			coefs[ialpha] = m.Coef
			dualGaps[ialpha] = m.CDResult.Gap
			nIters[ialpha] = m.CDResult.NIter
			wg.Done()
		}(ialpha, alpha)
	}
	wg.Wait()
	return
}

// LassoPath Compute lasso path with coordinate descent
func LassoPath(X, Y *mat.Dense, eps float64, NAlphas int, Alphas *[]float64, verbose, positive bool) (alphas []float64, coefs []*mat.Dense, dualGaps []float64, nIters []int) {
	alphas, coefs, dualGaps, nIters = EnetPath(X, Y, 1., eps, NAlphas, Alphas, verbose, positive)
	return
}
