package linearmodel

import (
	"errors"
	"fmt"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"

	//"gonum.org/v1/gonum/diff/fd"
	"math"
	"runtime"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

type float = float64

// Activation is borrowed from base package
type Activation = base.Activation

// LinearModel is a base struct for multioutput regressions
type LinearModel struct {
	FitIntercept, Normalize          bool
	XOffset, XScale, Coef, Intercept *mat.Dense
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
}

// RegularizedRegression is a common structure for ElasticNet,Lasso and Ridge
type RegularizedRegression struct {
	LinearRegression
	Solver              string
	SolverConfigure     func(base.Optimizer)
	Tol, Alpha, L1Ratio float64
	LossFunction        Loss
	ActivationFunction  Activation
	Options             LinFitOptions
}

// NewLinearRegression create a *LinearRegression with defaults
// implemented as mat.Dense.Solve
func NewLinearRegression() *LinearRegression {
	regr := &LinearRegression{}
	regr.FitIntercept = true

	return regr
}

// IsClassifier returns false for LinearRegression
func (*LinearRegression) IsClassifier() bool { return false }

// PredicterClone for LinearRegression
func (regr *LinearRegression) PredicterClone() base.Predicter {
	clone := *regr
	return &clone
}

// Fit fits Coef for a LinearRegression
func (regr *LinearRegression) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X0, Y0 := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	var X, Y, YOffset *mat.Dense
	X, Y, regr.XOffset, YOffset, regr.XScale = PreprocessData(X0, Y0, regr.FitIntercept, regr.Normalize, nil)
	// use least squares
	regr.Coef = &mat.Dense{}
	regr.Coef.Solve(X, Y)
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// GetNOutputs returns output columns number for Y to pass to predict
func (regr *LinearModel) GetNOutputs() int {
	_, nOutputs := regr.Coef.Dims()
	return nOutputs
}

// Fit fits Coef for a LinearRegression
func (regr *RegularizedRegression) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X0, Y0 := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	var X, Y, YOffset *mat.Dense
	X, Y, regr.XOffset, YOffset, regr.XScale = PreprocessData(X0, Y0, regr.FitIntercept, regr.Normalize, nil)
	opt := regr.Options
	opt.Tol = regr.Tol
	opt.Solver = regr.Solver
	opt.SolverConfigure = regr.SolverConfigure
	opt.Loss = regr.LossFunction
	opt.Activation = regr.ActivationFunction
	opt.Alpha = regr.Alpha
	opt.L1Ratio = regr.L1Ratio
	res := LinFit(X, Y, &opt)
	regr.Coef = res.Theta
	regr.LinearModel.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// Predict predicts y for X using Coef
func (regr *LinearRegression) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, regr.GetNOutputs(), nil)
	}
	regr.DecisionFunction(X, Y)
	return base.FromDense(Ymutable, Y)
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

// IsClassifier returns false for SGDRegressor
func (*SGDRegressor) IsClassifier() bool { return false }

// PredicterClone for SGDRegressor
func (regr *SGDRegressor) PredicterClone() base.Predicter {
	clone := *regr
	return &clone
}

// Fit learns Coef
func (regr *SGDRegressor) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X0, y0 := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	var X, Y, YOffset *mat.Dense
	X, Y, regr.XOffset, YOffset, regr.XScale = PreprocessData(X0, y0, regr.FitIntercept, regr.Normalize, nil)
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
			if grad == nil {
				grad = make([]float64, len(coef))
			}
			// X dot Ydiff+ alpha*l1ratio*sign+alpha*(1-l1ratio)*coef
			tmp := mat.NewDense(nSamples, 1, nil)
			regr.Coef.SetCol(o, coef)
			tmp.Mul(X, regr.Coef.ColView(o)) // X dot coef
			tmp.Sub(tmp, Y.ColView(o))       // Ydiff
			gradmat := mat.NewDense(nFeatures, 1, nil)
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
		initialcoefs := make([]float, nFeatures)
		/*for j := 0; j < nFeatures; j++ {
			initialcoefs[j] = rand.Float64()
		}*/
		settings := &optimize.Settings{}
		//settings.FunctionThreshold = regr.Tol * regr.Tol * 1e-4
		//settings.GradientThreshold = 1.e-6
		//settings.FuncEvaluations = 100000
		/*  settings.FunctionConverge.Iterations = 1000
		 */
		if regr.NJobs <= 0 {
			settings.Concurrent = runtime.NumCPU()
		} else {
			settings.Concurrent = regr.NJobs
		}

		// printer := NewPrinter()
		// printer.HeadingInterval = 1
		// settings.Recorder = printer

		method := regr.Method
		res, err := optimize.Minimize(p, initialcoefs, settings, method)
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
func (regr *SGDRegressor) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, regr.GetNOutputs(), nil)
	}
	regr.DecisionFunction(X, Y)
	return base.FromDense(Ymutable, Y)
}

func unused(...interface{}) {}

// LinFitOptions are options for LinFit
type LinFitOptions struct {
	Epochs, MiniBatchSize int
	Tol                   float64
	Solver                string
	SolverConfigure       func(base.Optimizer)
	// Alpha is regularization factor for Ridge,Lasso
	Alpha float64
	// L1Ratio is the part of L1 regularization 0 for ridge,1 for Lasso
	L1Ratio                             float64
	Loss                                Loss
	Activation                          Activation
	GOMethodCreator                     func() optimize.Method
	ThetaInitializer                    func(Theta *mat.Dense)
	Recorder                            optimize.Recorder
	PerOutputFit                        bool
	DisableRegularizationOfFirstFeature bool
}

// LinFitResult is the result or LinFit
type LinFitResult struct {
	Converged bool
	RMSE, J   float64
	Epoch     int
	Theta     *mat.Dense
}

func initRecorder(recorder optimize.Recorder) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	err = errors.New("no recorcder")
	return recorder.Init()
}

// LinFit is an internal helper to fit linear regressions
func LinFit(X, Ytrue *mat.Dense, opts *LinFitOptions) *LinFitResult {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()
	if opts.GOMethodCreator == nil && opts.Solver == "" {
		opts.GOMethodCreator = func() optimize.Method { return &optimize.LBFGS{} }
	}
	if opts.GOMethodCreator != nil {
		opts.PerOutputFit = true
		return LinFitGOM(X, Ytrue, opts)
	}

	thetaSlice := make([]float64, nFeatures*nOutputs)
	thetaSliceBest := make([]float64, nFeatures*nOutputs)

	Theta := mat.NewDense(nFeatures, nOutputs, thetaSlice)
	gradSlice := make([]float64, nFeatures*nOutputs)
	grad := mat.NewDense(nFeatures, nOutputs, gradSlice)

	if opts.ThetaInitializer != nil {
		opts.ThetaInitializer(Theta)
	} else {
		Theta.Apply(func(i, j int, v float64) float64 {
			return 0.01 * rand.Float64()
		}, Theta)
	}

	var (
		miniBatchStart = 0
		miniBatchSize  = 200
	)
	if opts.MiniBatchSize > 0 {
		miniBatchSize = opts.MiniBatchSize
	} else {
		miniBatchSize = int(math.Max(1., math.Min(100., math.Sqrt(float64(nSamples)))))
	}
	miniBatchSize = int(math.Max(1., math.Min(float64(nSamples), float64(miniBatchSize))))
	if opts.Loss == nil {
		opts.Loss = SquareLoss
	}
	if opts.Activation == nil {
		opts.Activation = base.Identity{}
	}

	YpredMini := mat.NewDense(miniBatchSize, nOutputs, nil)
	YdiffMini := mat.NewDense(miniBatchSize, nOutputs, nil)

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)

	s := base.NewSolver(opts.Solver)()
	if opts.SolverConfigure != nil {
		opts.SolverConfigure(s)
	}
	s.SetTheta(Theta)
	rmse := math.Inf(1.)
	J := math.Inf(1.)
	JBest := math.Inf(1.)
	converged := false
	start := time.Now()
	if opts.Epochs <= 0 {
		opts.Epochs = 1e6 / nSamples
	}
	var epoch int
	var hasRecorder = initRecorder(opts.Recorder) == nil
	if hasRecorder {
		opts.Recorder.Record(
			&optimize.Location{X: thetaSlice, F: J, Gradient: gradSlice},
			optimize.InitIteration,
			&optimize.Stats{MajorIterations: epoch, FuncEvaluations: epoch, GradEvaluations: epoch, Runtime: time.Since(start)})
	}
	for epoch = 1; epoch <= opts.Epochs && !converged; epoch++ {
		shuffler := preprocessing.NewShuffler()
		Xs, Ys := shuffler.FitTransform(X, Ytrue)
		for miniBatch := 0; miniBatch*miniBatchSize < nSamples; miniBatch++ {
			miniBatchStart = miniBatch * miniBatchSize
			miniBatchEnd := miniBatchStart + miniBatchSize
			if miniBatchEnd > nSamples {
				miniBatchEnd = nSamples
			}
			miniBatchRows := miniBatchEnd - miniBatchStart

			opts.Loss(
				Ys.Slice(miniBatchStart, miniBatchEnd, 0, nOutputs),
				Xs.Slice(miniBatchStart, miniBatchEnd, 0, nFeatures),
				Theta,
				YpredMini.Slice(0, miniBatchRows, 0, nOutputs).(*mat.Dense),
				YdiffMini.Slice(0, miniBatchRows, 0, nOutputs).(*mat.Dense),
				grad,
				opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
			s.UpdateParams(grad)
		}
		J = opts.Loss(
			Ytrue,
			X,
			Theta,
			Ypred,
			Ydiff,
			grad,
			opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
		if J < JBest {
			JBest = J
			copy(thetaSliceBest, thetaSlice)
		}
		rmse = math.Sqrt(metrics.MeanSquaredError(Ytrue, Ypred, nil, "").At(0, 0))

		converged = math.Sqrt(rmse) < opts.Tol
		//fmt.Println(epoch, J)
		if hasRecorder {
			opts.Recorder.Record(
				&optimize.Location{X: thetaSlice, F: J, Gradient: gradSlice},
				optimize.InitIteration,
				&optimize.Stats{MajorIterations: epoch, FuncEvaluations: epoch, GradEvaluations: epoch, Runtime: time.Since(start)})
		}
	}
	J = JBest
	Theta = mat.NewDense(nFeatures, nOutputs, thetaSliceBest)
	return &LinFitResult{Converged: converged, RMSE: rmse, J: J, Epoch: epoch, Theta: Theta}
}

// LinFitGOM fits a regression with a gonum/optimizer Method
func LinFitGOM(X, Ytrue *mat.Dense, opts *LinFitOptions) *LinFitResult {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Ytrue.Dims()

	if opts.Loss == nil {
		opts.Loss = SquareLoss
	}
	if opts.Activation == nil {
		opts.Activation = base.Identity{}
	}

	converged := false
	if opts.Epochs <= 0 {
		opts.Epochs = 4e6 / nSamples
	}
	fSettings := func() *optimize.Settings {
		settings := &optimize.Settings{}
		settings.Recorder = opts.Recorder
		settings.GradientThreshold = 1e-12
		settings.FuncEvaluations = opts.Epochs
		settings.Concurrent = runtime.NumCPU()
		//settings.Recorder = optimize.NewPrinter()
		return settings
	}

	theta := make([]float64, nFeatures*nOutputs)
	thetaM := mat.NewDense(nFeatures, nOutputs, theta)
	for j := 0; j < len(theta); j++ {
		theta[j] = 0.01 * rand.NormFloat64()
	}
	var ret *optimize.Result
	var err error
	rmse := 0.
	epoch := 0
	converged = true
	if opts.PerOutputFit {
		type fitOutputRes struct {
			o   int
			ret *optimize.Result
			err error
		}
		chanret := make(chan fitOutputRes, nOutputs)

		fitOutput := func(o int, chanret chan fitOutputRes) {
			Ypred := mat.NewDense(nSamples, 1, nil)
			Ydiff := mat.NewDense(nSamples, 1, nil)
			thetao := make([]float64, nFeatures)
			p := optimize.Problem{
				Func: func(thetao []float64) float64 {
					J := opts.Loss(Ytrue.ColView(o), X, mat.NewDense(nFeatures, 1, thetao), Ypred, Ydiff, nil, opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
					return J
				},
				Grad: func(grad, thetao []float64) {
					if grad == nil {
						grad = make([]float64, len(thetao))
					}
					if len(thetao) != len(grad) {
						panic("incorrect size of the gradient")
					}

					opts.Loss(Ytrue.ColView(o), X, mat.NewDense(nFeatures, 1, thetao), Ypred, Ydiff, mat.NewDense(nFeatures, 1, grad), opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
				},
			}
			mat.Col(thetao, o, thetaM)
			ret, err := optimize.Minimize(p, thetao, fSettings(), opts.GOMethodCreator())
			//fmt.Printf("output %d F:%v Grad:%v Status:%s\n", o, ret.F, mat.Norm(mat.NewVecDense(nFeatures, ret.Gradient), math.Inf(1)), ret.Status)
			chanret <- fitOutputRes{o: o, ret: ret, err: err}
		}
		for o1 := 0; o1 < nOutputs; o1++ {
			go fitOutput(o1, chanret)
		}

		for o1 := 0; o1 < nOutputs; o1++ {
			foret := <-chanret
			ret := foret.ret
			thetaM.SetCol(foret.o, ret.X)
			rmse += ret.F
			epoch += ret.FuncEvaluations
			converged = converged && ret.Status != optimize.Failure
		}
		rmse = math.Sqrt(rmse) / float64(nOutputs)

	} else {

		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		Ydiff := mat.NewDense(nSamples, nOutputs, nil)
		p := optimize.Problem{
			Func: func(theta []float64) float64 {

				J := opts.Loss(Ytrue, X, mat.NewDense(nFeatures, nOutputs, theta), Ypred, Ydiff, nil, opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
				return J
			},
			Grad: func(grad, theta []float64) {
				if grad == nil {
					grad = make([]float64, len(theta))
				}
				opts.Loss(Ytrue, X, mat.NewDense(nFeatures, nOutputs, theta), Ypred, Ydiff, mat.NewDense(nFeatures, nOutputs, grad), opts.Alpha, opts.L1Ratio, nSamples, opts.Activation, opts.DisableRegularizationOfFirstFeature)
			},
		}
		ret, err = optimize.Minimize(p, theta, fSettings(), opts.GOMethodCreator())
		copy(theta, ret.X)
		rmse = mat.Norm(Ydiff, 2) / float64(nOutputs)
		epoch = ret.FuncEvaluations
		converged = ret.Status != optimize.Failure
	}
	//fmt.Printf("ret:%#v\nstatus:%s\n", ret, ret.Status)
	converged = err == nil
	return &LinFitResult{Converged: converged, RMSE: rmse, Epoch: epoch, Theta: thetaM}
}

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

// DecisionFunction fills Y with X dot Coef+Intercept
func (regr *LinearModel) DecisionFunction(X mat.Matrix, Ymutable mat.Mutable) {
	Y := base.ToDense(Ymutable)
	Y.Mul(X, regr.Coef)
	Y.Apply(func(j int, o int, v float64) float64 {

		return v + regr.Intercept.At(0, o)
	}, Y)
	base.FromDense(Ymutable, Y)
}

// Score returns R2Score between Y and X dot Coef+Intercept
func (regr *LinearModel) Score(X, Y mat.Matrix) float64 {
	Yd := base.ToDense(Y)
	Ypred := &mat.Dense{}
	regr.DecisionFunction(X, Ypred)
	return metrics.R2Score(Yd, Ypred, nil, "").At(0, 0)
}

// PreprocessData center and normalize data
func PreprocessData(X, Y *mat.Dense, FitIntercept, Normalize bool, SampleWeight *mat.VecDense) (Xout, Yout, XOffset, YOffset, XScale *mat.Dense) {
	Xmat := X.RawMatrix()
	Ymat := Y.RawMatrix()
	XOffset = mat.NewDense(1, Xmat.Cols, nil)
	XOffsetmat := XOffset.RawMatrix()
	YOffset = mat.NewDense(1, Ymat.Cols, nil)
	YOffsetmat := YOffset.RawMatrix()
	XScale = mat.NewDense(1, Xmat.Cols, nil)
	XScalemat := XScale.RawMatrix()
	for i := range XScalemat.Data[0:XScalemat.Cols] {
		XScalemat.Data[i] = 1
	}
	if !FitIntercept && !Normalize {
		Xout, Yout = X, Y
		return
	}
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()

	Yout = Y
	base.Parallelize(-1, Xmat.Cols, func(th, start, end int) {
		for feature := start; feature < end; feature++ {
			xcol := X.ColView(feature)
			mean := 0.
			if FitIntercept {
				mean = mat.Sum(xcol) / float64(Xmat.Rows)

				for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {
					Xoutmat.Data[jXout+feature] = Xmat.Data[jX+feature] - mean
				}
			} else {
				Xout.Copy(X)
			}
			scale := 1.
			if FitIntercept && Normalize {
				scale = mat.Norm(Xout.ColView(feature), 2)
				if scale != 0. {
					for jXout := 0; jXout < Xoutmat.Rows*Xoutmat.Stride; jXout = jXout + Xoutmat.Stride {
						Xoutmat.Data[jXout+feature] /= scale
					}
				}

			}
			XOffsetmat.Data[feature] = mean
			XScalemat.Data[feature] = scale

		}
	})
	if FitIntercept {
		Yout = mat.NewDense(Ymat.Rows, Ymat.Cols, nil)
		base.Parallelize(-1, Ymat.Cols, func(th, start, end int) {
			for output := start; output < end; output++ {
				ycol := Y.ColView(output)
				mean := mat.Sum(ycol) / float64(Ymat.Rows)
				YOffsetmat.Data[output] = mean
				Youtmat := Yout.RawMatrix()
				for jY, jYout := 0, 0; jY < Ymat.Rows*Ymat.Stride; jY, jYout = jY+Ymat.Stride, jYout+Youtmat.Stride {
					Youtmat.Data[jYout+output] = Ymat.Data[jY+output] - mean

				}
			}
		})
	} else {
		Yout = Y
	}
	// fmt.Printf("debug\nXout\n%.3f\nYout\n%.3f\nXOffset\n%.3f\nYOffset\n%.3f\nXScale\n%.3f\n",
	// 	mat.Formatted(Xout), mat.Formatted(Yout), mat.Formatted(XOffset), mat.Formatted(YOffset), mat.Formatted(XScale))
	return
}
