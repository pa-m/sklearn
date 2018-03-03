package linearModel

import (
	"fmt"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	//"gonum.org/v1/gonum/diff/fd"
	"math"
	"math/rand"
	"runtime"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

type float = float64

// LinearModel is a base struct for multioutput regressions
type LinearModel struct {
	FitIntercept, Normalize          bool
	XOffset, XScale, Coef, Intercept *mat.Dense
}

// Regressor is the common interface for all regressors
type Regressor interface {
	Fit(X, Y *mat.Dense) Regressor
	Predict(X, Y *mat.Dense)
	Score(X, T *mat.Dense) float64
}

// LinearRegression ia Ordinary least squares Linear Regression.
// Parameters
// ----------
// fitIntercept : boolean, optional, default True
//     whether to calculate the intercept for this model. If set
//     to False, no intercept will be used in calculations
//     (e.g. data is expected to be already centered).
// normalize : boolean, optional, default False
//     This parameter is ignored when ``fitIntercept`` is set to False.
//     If True, the regressors X will be normalized before regression by
//     subtracting the mean and dividing by the l2-norm.
//     If you wish to standardize, please use
//     :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
//     an estimator with ``normalize=False``.
// ----------
// coef : array, shape (nFeatures, ) or (nTargets, nFeatures)
//     Estimated coefficients for the linear regression problem.
//     If multiple targets are passed during the fit (y 2D), this
//     is a 2D array of shape (nTargets, nFeatures), while if only
//     one target is passed, this is a 1D array of length nFeatures.
// intercept : array
//     Independent term in the linear model.
type LinearRegression struct {
	LinearModel
	Optimizer           base.Optimizer
	Tol, Alpha, L1Ratio float64
	LossFunction        Loss
	ActivationFunction  Activation
	Options             LinFitOptions
}

// NewLinearRegression create a *LinearRegression with defaults
// implemented as a per-output optimization of (possibly regularized) square-loss a base.Optimizer (defaults to Adam)
func NewLinearRegression() *LinearRegression {
	regr := &LinearRegression{Tol: 1e-6}
	regr.Optimizer = base.NewAdamOptimizer()
	regr.FitIntercept = true
	regr.Normalize = false
	regr.ActivationFunction = Identity{}
	regr.LossFunction = SquareLoss
	return regr
}

// Fit fits Coef for a LinearRegression
func (regr *LinearRegression) Fit(X0, Y0 *mat.Dense) Regressor {
	X := mat.DenseCopyOf(X0)
	regr.XOffset, regr.XScale = preprocessing.DenseNormalize(X, regr.FitIntercept, regr.Normalize)
	Y := mat.DenseCopyOf(Y0)
	YOffset, _ := preprocessing.DenseNormalize(Y, regr.FitIntercept, false)
	opt := regr.Options
	opt.Tol = regr.Tol
	opt.Solver = regr.Optimizer
	opt.Loss = regr.LossFunction
	opt.Activation = regr.ActivationFunction
	res := LinFit(X, Y, &opt)
	regr.Coef = res.Theta
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// Predict predicts y for X using Coef
func (regr *LinearRegression) Predict(X, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)

}

// NewRidge creates a *Ridge with defaults
func NewRidge() *LinearRegression {
	regr := NewLinearRegression()
	regr.Alpha = 1.
	regr.L1Ratio = 0.
	return regr
}

//NewLasso creates a *Lasso with defaults
func NewLasso() *LinearRegression {
	regr := NewLinearRegression()
	regr.Alpha = 1e-3
	regr.L1Ratio = 1.
	return regr
}

// SGDRegressor base struct
// should  be named GonumOptimizeRegressor
// implemented as a per-output optimization of (possibly regularized) square-loss with gonum/optimize methods
type SGDRegressor struct {
	LinearModel
	Tol, Alpha, L1Ratio float
	NJobs               int
	Method              optimize.Method
}

// NewSGDRegressor creates a *SGDRegressor with defaults
func NewSGDRegressor() *SGDRegressor {
	regr := &SGDRegressor{Tol: 1e-4, Alpha: 0.0001, L1Ratio: 0.15, NJobs: 1, Method: &optimize.LBFGS{}}
	regr.FitIntercept = true
	//regr.RegressorMixin1.Predicter = regr
	return regr
}

// Fit learns Coef
func (regr *SGDRegressor) Fit(X0, y0 *mat.Dense) Regressor {
	X := mat.DenseCopyOf(X0)
	regr.XOffset, regr.XScale = preprocessing.DenseNormalize(X, regr.FitIntercept, regr.Normalize)
	Y := mat.DenseCopyOf(y0)
	YOffset, _ := preprocessing.DenseNormalize(Y, regr.FitIntercept, false)
	// begin use gonum gradientDescent
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Y.Dims()

	loss := func(coefSlice []float, o int) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		regr.Coef.SetCol(o, coefSlice)
		tmp := mat.NewDense(nSamples, 1, nil)
		// e will be sum of squares of errors
		tmp.Mul(X, regr.Coef.ColView(o))
		tmp.Sub(tmp, Y.ColView(o))
		tmp.MulElem(tmp, tmp)
		e := mat.Sum(tmp) / 2. / float(nSamples)

		L1 := 0.
		L2 := 0.
		if regr.Alpha > 0. {
			// compute regularization term
			alphaL1 := regr.Alpha * regr.L1Ratio / 2. / float64(nSamples)
			alphaL2 := regr.Alpha * (1. - regr.L1Ratio) / 2. / float64(nSamples)

			for j := 0; j < nFeatures; j++ {
				c := regr.Coef.At(j, o)
				L1 += alphaL1 * math.Abs(c)
				L2 += alphaL2 * c * c
			}
		}
		loss := (e + L1 + L2)
		//fmt.Printf("%T loss %g\n", regr.Method, loss)
		return loss
	}
	regr.Coef = mat.NewDense(nFeatures, nOutputs, nil)
	for o := 0; o < nOutputs; o++ {
		p := optimize.Problem{}
		p.Func = func(coefSlice []float64) float64 { return loss(coefSlice, o) }
		/* gradient from diff/fd is not good enough for multioutput regression
		p.Grad = func(grad, coef []float) {
			h := 1e-6

			settings := &fd.Settings{}
			settings.Concurrent = true
			settings.Step = h
			fd.Gradient(grad, p.Func, coef, settings)

		}*/
		p.Grad = func(grad, coef []float) {
			// X dot Ydiff+ alpha*l1ratio*sign+alpha*(1-l1ratio)*coef
			tmp := mat.NewDense(nSamples, 1, nil)
			regr.Coef.SetCol(o, coef)
			tmp.Mul(X, regr.Coef.ColView(o)) // X dot coef
			tmp.Sub(tmp, Y.ColView(o))       // Ydiff
			gradmat := mat.NewDense(nFeatures, 1, nil)
			chkdims(".", gradmat, X.T(), tmp)
			gradmat.Mul(X.T(), tmp) // X dot Ydiff
			al1 := regr.Alpha * regr.L1Ratio / float(nSamples)
			al2 := regr.Alpha * (1. - regr.L1Ratio) / float(nSamples)
			sgn := func(x float) float {
				if x < 0. {
					return -1.
				}
				if x > 0. {
					return 1.
				}
				return 0.
			}
			gradmat.Apply(func(j int, o int, gradj float64) float64 {
				grad[j] = gradj + al1*sgn(gradj) + al2*gradj
				return grad[j]
			}, gradmat)
		}
		initialcoefs := make([]float, nFeatures, nFeatures)
		/*for j := 0; j < nFeatures; j++ {
			initialcoefs[j] = rand.Float64()
		}*/
		settings := optimize.DefaultSettings()
		//settings.FunctionThreshold = regr.Tol * regr.Tol * 1e-4
		//settings.GradientThreshold = 1.e-6
		//settings.FuncEvaluations = 100000
		/*  settings.FunctionConverge.Iterations = 1000
		 */
		settings.FunctionConverge = nil
		if regr.NJobs <= 0 {
			settings.Concurrent = runtime.NumCPU()
		} else {
			settings.Concurrent = regr.NJobs
		}

		// printer := NewPrinter()
		// printer.HeadingInterval = 1
		// settings.Recorder = printer

		method := regr.Method
		res, err := optimize.Local(p, initialcoefs, settings, method)
		unused(err)
		//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
		// if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {
		//
		// 	fmt.Println(err)
		// }
		regr.Coef.SetCol(o, res.X)
	}

	// end use gonum gradient gradientDescent
	regr.setIntercept(regr.XOffset, YOffset, regr.XScale)

	return regr
}

// Predict predicts y from X using Coef
func (regr *SGDRegressor) Predict(X, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)
	return
}

func unused(...interface{}) {}

// LinFitOptions are options for LinFit
type LinFitOptions struct {
	Epochs, MiniBatchSize int
	Tol                   float64
	Solver                base.Optimizer
	// Alpha is regularization factor for Ridge,Lasso
	Alpha float64
	// L1Ratio is the part of L1 regularization 0 for ridge,1 for Lasso
	L1Ratio    float64
	Loss       Loss
	Activation Activation
	GOMethod   optimize.Method
}

// LinFitResult is the result or LinFit
type LinFitResult struct {
	Converged bool
	RMSE      float64
	Epoch     int
	Theta     *mat.Dense
}

// LinFit is an internal helper to fit linear regressions
func LinFit(X, Ytrue *mat.Dense, opts *LinFitOptions) *LinFitResult {
	if opts.GOMethod != nil {
		return LinFitGOM(X, Ytrue, opts)
	}
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()

	Theta := mat.NewDense(nFeatures, nOutputs, nil)
	Theta.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()
	}, Theta)
	var (
		miniBatchStart = 0
		miniBatchSize  = 200
	)
	if opts.MiniBatchSize > 0 {
		miniBatchSize = opts.MiniBatchSize
	}
	miniBatchSize = int(math.Max(1, math.Min(100., math.Sqrt(float64(nSamples)))))
	if opts.Loss == nil {
		opts.Loss = SquareLoss
	}
	if opts.Activation == nil {
		opts.Activation = Identity{}
	}

	YpredMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	YdiffMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	grad := mat.NewDense(nFeatures, nOutputs, nil)

	Ypred := mat.NewDense(nSamples, nOutputs, nil)

	s := opts.Solver
	s.SetTheta(Theta)
	var timeStep uint64
	rmse := math.Inf(1)
	converged := false
	if opts.Epochs <= 0 {
		opts.Epochs = 4e6 / nSamples
	}
	J := math.Inf(1)
	var epoch int
	for epoch = 1; epoch <= opts.Epochs && !converged; epoch++ {
		base.DenseShuffle(X, Ytrue)
		for miniBatch := 0; miniBatch*miniBatchSize < nSamples; miniBatch++ {
			miniBatchStart = miniBatch * miniBatchSize
			miniBatchEnd := miniBatchStart + miniBatchSize
			if miniBatchEnd > nSamples {
				miniBatchEnd = nSamples
			}
			miniBatchRows := miniBatchEnd - miniBatchStart
			Xmini := (X.Slice(miniBatchStart, miniBatchEnd, 0, nFeatures))
			YtrueMini := (Ytrue.Slice(miniBatchStart, miniBatchEnd, 0, nOutputs))

			J = opts.Loss(YtrueMini, Xmini, Theta, YpredMini.Slice(0, miniBatchRows, 0, nOutputs).(*mat.Dense), YdiffMini.Slice(0, miniBatchRows, 0, nOutputs).(*mat.Dense), grad, opts.Alpha, opts.L1Ratio, miniBatchSize, opts.Activation)

			s.UpdateParams(grad)
			unused(J)
			// if epoch >= opts.Epochs {
			// if miniBatch == 0 && epoch%10 == 0 {
			// 	fmt.Printf("%d %d LinFit J=%g grad0:%v\n", epoch, miniBatch, J, grad.ColView(0))
			// }

			//}
		}

		//d := func(X mat.Matrix) string { r, c := X.Dims(); return fmt.Sprintf("%d,%d", r, c) }
		Ypred.Mul(X, Theta)
		switch opts.Activation.(type) {
		case Identity:
		default:
			Ypred.Apply(func(j, o int, xtheta float64) float64 { return opts.Activation.F(xtheta) }, Ypred)
		}
		rmse = math.Sqrt(metrics.MeanSquaredError(Ytrue, Ypred, nil, "").At(0, 0))

		converged = math.Sqrt(rmse) < opts.Tol
		timeStep = s.GetTimeStep()

	}
	unused(timeStep, fmt.Println)
	return &LinFitResult{Converged: converged, RMSE: rmse, Epoch: epoch, Theta: Theta}
}

// LinFitGOM fits a regression with a gonum/optimizer Method
func LinFitGOM(X, Ytrue *mat.Dense, opts *LinFitOptions) *LinFitResult {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()

	if opts.Loss == nil {
		opts.Loss = SquareLoss
	}
	if opts.Activation == nil {
		opts.Activation = Identity{}
	}

	gradslice := make([]float64, nFeatures*nOutputs, nFeatures*nOutputs)
	grad := mat.NewDense(nFeatures, nOutputs, gradslice)

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)

	rmse := math.Inf(1)
	converged := false
	if opts.Epochs <= 0 {
		opts.Epochs = 4e6 / nSamples
	}
	p := optimize.Problem{
		Func: func(theta []float64) float64 {

			J := opts.Loss(Ytrue, X, mat.NewDense(nFeatures, nOutputs, theta), Ypred, Ydiff, grad, opts.Alpha, opts.L1Ratio, nSamples, opts.Activation)
			return J
		},
		Grad: func(gradSlice, theta []float64) {
			opts.Loss(Ytrue, X, mat.NewDense(nFeatures, nOutputs, theta), Ypred, Ydiff, grad, opts.Alpha, opts.L1Ratio, nSamples, opts.Activation)
			copy(gradSlice, gradslice)
		},
	}
	settings := optimize.DefaultSettings()
	settings.Recorder = nil
	settings.GradientThreshold = 1e-12
	settings.FunctionConverge = nil
	settings.FuncEvaluations = opts.Epochs
	settings.FunctionThreshold = opts.Tol * opts.Tol
	//settings.Recorder = optimize.NewPrinter()
	theta := make([]float64, nFeatures*nOutputs, nFeatures*nOutputs)
	for j := 0; j < len(theta); j++ {
		theta[j] = 0.01 * rand.NormFloat64()
	}
	ret, err := optimize.Local(p, theta, settings, opts.GOMethod)

	//fmt.Printf("ret:%#v\nstatus:%s\n", ret, ret.Status)
	copy(theta, ret.X)
	Theta := mat.NewDense(nFeatures, nOutputs, theta)
	converged = err == nil
	Ypred.Mul(X, Theta)
	switch opts.Activation.(type) {
	case Identity:
	default:
		Ypred.Apply(func(j, o int, xtheta float64) float64 { return opts.Activation.F(xtheta) }, Ypred)
	}
	rmse = math.Sqrt(metrics.MeanSquaredError(Ytrue, Ypred, nil, "").At(0, 0))
	epoch := ret.FuncEvaluations
	return &LinFitResult{Converged: converged, RMSE: rmse, Epoch: epoch, Theta: Theta}
}

// SetIntercept
// def _set_intercept(self, X_offset, y_offset, X_scale):
// 		"""Set the intercept_
// 		"""
// 		if self.fit_intercept:
// 				self.coef_ = self.coef_ / X_scale
// 				self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
// 		else:
// 				self.intercept_ = 0.
func (regr *LinearModel) setIntercept(XOffset, YOffset, XScale mat.Matrix) {
	_, nOutputs := regr.Coef.Dims()
	if regr.Intercept == nil {
		regr.Intercept = mat.NewDense(1, nOutputs, nil)
	}
	regr.Coef.Apply(func(j, o int, coef float64) float64 { return coef / XScale.At(0, j) }, regr.Coef)
	if regr.FitIntercept {
		regr.Intercept.Mul(XOffset, regr.Coef)
		regr.Intercept.Sub(YOffset, regr.Intercept)
	}
}

func dims(mats ...mat.Matrix) string {
	s := ""
	for _, m := range mats {
		r, c := m.Dims()
		s = fmt.Sprintf("%s %d,%d", s, r, c)
	}
	return s
}

func chkdims(op string, R, X, Y mat.Matrix) {
	rx, cx := X.Dims()
	ry, cy := Y.Dims()
	rr, cr := R.Dims()
	switch op {
	case "+", "-", "*", "/":
		if rx != ry || cx != cy || rr != rx || cr != cx {
			panic(fmt.Errorf("%s %s", op, dims(R, X, Y)))
		}
	case ".":
		if cx != ry || rr != rx || cr != cy {
			panic(fmt.Errorf("%s %s", op, dims(R, X, Y)))
		}
	}
}

// DecisionFunction fills Y with X dot Coef+Intercept
func (regr *LinearModel) DecisionFunction(X, Y *mat.Dense) {
	chkdims(".", Y, X, regr.Coef)
	Y.Mul(X, regr.Coef)
	Y.Apply(func(j int, o int, v float64) float64 {

		return v + regr.Intercept.At(0, o)
	}, Y)
}

// Score returns R2Score between Y and X dot Coef+Intercept
func (regr *LinearModel) Score(X, Y *mat.Dense) float64 {
	nSamples, nOutputs := Y.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	regr.DecisionFunction(X, Ypred)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
