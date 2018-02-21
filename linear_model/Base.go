package linear_model

import (
	"fmt"
	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/diff/fd"
	. "gonum.org/v1/gonum/optimize"
	"math"
	"math/rand"
	"os"
	"runtime"
)

type float = float64

type LinearModel struct {
	X_offset_, X_scale_     []float
	Coef_                   []float
	Intercept_              float
	FitIntercept, Normalize bool
}

// """
// Ordinary least squares Linear Regression.
// Parameters
// ----------
// fit_intercept : boolean, optional, default True
//     whether to calculate the intercept for this model. If set
//     to False, no intercept will be used in calculations
//     (e.g. data is expected to be already centered).
// normalize : boolean, optional, default False
//     This parameter is ignored when ``fit_intercept`` is set to False.
//     If True, the regressors X will be normalized before regression by
//     subtracting the mean and dividing by the l2-norm.
//     If you wish to standardize, please use
//     :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
//     an estimator with ``normalize=False``.
// ----------
// coef_ : array, shape (n_features, ) or (n_targets, n_features)
//     Estimated coefficients for the linear regression problem.
//     If multiple targets are passed during the fit (y 2D), this
//     is a 2D array of shape (n_targets, n_features), while if only
//     one target is passed, this is a 1D array of length n_features.
// intercept_ : array
//     Independent term in the linear model.

type LinearRegression struct {
	LinearModel
	base.RegressorMixin
	Tol   float
	NJobs int
}

func NewLinearRegression() *LinearRegression {
	self := &LinearRegression{Tol: 1e-6, NJobs: 1}
	self.LinearModel.FitIntercept = true
	self.RegressorMixin.Predicter = self
	return self
}

func (self *LinearRegression) Fit(X0 [][]float, y0 []float) *LinearRegression {
	var n_features = len(X0[0])
	var X, y, X_offset_, y_offset_, X_scale_ = preprocess_data(
		X0, y0, self.FitIntercept, self.Normalize)
	self.X_offset_ = X_offset_
	self.X_scale_ = X_scale_
	loss := func(coef_ []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, n_features, n_features)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef_, Xi))
			e += e1 * e1
			//fmt.Printf("coef_ %v yi %g yp %g e1 %g e %g\n", coef_, y[i], yp, e1, e)
		}
		return e
	}
	p := Problem{}
	p.Func = loss
	p.Grad = func(grad, coef_ []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef_, settings)

	}
	initialcoefs_ := make([]float, n_features, n_features)
	for j := 0; j < n_features; j++ {
		initialcoefs_[j] = rand.Float64()
	}
	settings := DefaultSettings()
	settings.FunctionThreshold = self.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if self.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = self.NJobs
	}

	// printer := NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &CG{}
	res, err := Local(p, initialcoefs_, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	self.Coef_ = res.X
	self._set_intercept(X_offset_, y_offset_, X_scale_)

	return self
}

func (self *LinearRegression) Predict(X [][]float) (y_mean []float) {
	y_mean = self.DecisionFunction(X)
	return
}

// ----
type Ridge struct {
	LinearModel
	base.RegressorMixin
	Alpha, Tol float
	NJobs      int
}

func NewRidge() *Ridge {
	self := &Ridge{Alpha: 1., Tol: 1e-3, NJobs: 1}
	self.LinearModel.FitIntercept = true
	self.RegressorMixin.Predicter = self
	return self
}

func (self *Ridge) Fit(X0 [][]float, y0 []float) *Ridge {
	if self.Normalize {
		fmt.Fprintf(os.Stderr, "Ridge don't work with Normalize. Reverting Normalize to False")
		self.Normalize = false
	}
	var n_features = len(X0[0])
	var X, y, X_offset_, y_offset_, X_scale_ = preprocess_data(
		X0, y0, self.FitIntercept, self.Normalize)
	self.X_offset_ = X_offset_
	self.X_scale_ = X_scale_
	loss := func(coef_ []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, n_features, n_features)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef_, Xi))
			e += e1 * e1
			//fmt.Printf("coef_ %v yi %g yp %g e1 %g e %g\n", coef_, y[i], yp, e1, e)
		}
		coef2 := make([]float, n_features, n_features)
		floats.MulTo(coef2, coef_, coef_)
		e = e/float(len(X)) + self.Alpha*floats.Sum(coef2)/float(len(coef_))
		return e
	}
	p := Problem{}
	p.Func = loss
	p.Grad = func(grad, coef_ []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef_, settings)

	}
	initialcoefs_ := make([]float, n_features, n_features)
	for j := 0; j < n_features; j++ {
		initialcoefs_[j] = rand.Float64()
	}
	settings := DefaultSettings()
	settings.FunctionThreshold = self.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if self.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = self.NJobs
	}

	// printer := NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &CG{}
	res, err := Local(p, initialcoefs_, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	self.Coef_ = res.X
	self._set_intercept(X_offset_, y_offset_, X_scale_)

	return self
}

func (self *Ridge) Predict(X [][]float) (y_mean []float) {
	y_mean = self.DecisionFunction(X)
	return
}

// ---
type Lasso struct {
	LinearModel
	base.RegressorMixin
	Alpha, Tol float
	NJobs      int
}

func NewLasso() *Lasso {
	self := &Lasso{Alpha: 1., Tol: 1e-4, NJobs: 1}
	self.LinearModel.FitIntercept = true
	self.RegressorMixin.Predicter = self
	return self
}

func (self *Lasso) Fit(X0 [][]float, y0 []float) *Lasso {
	if self.Normalize {
		fmt.Fprintf(os.Stderr, "Lasso don't work with Normalize. Reverting Normalize to False")
		self.Normalize = false
	}
	var n_features = len(X0[0])
	var X, y, X_offset_, y_offset_, X_scale_ = preprocess_data(
		X0, y0, self.FitIntercept, self.Normalize)
	self.X_offset_ = X_offset_
	self.X_scale_ = X_scale_
	squares := func(coef_ []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, n_features, n_features)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef_, Xi))
			e += e1 * e1
			//fmt.Printf("coef_ %v yi %g yp %g e1 %g e %g\n", coef_, y[i], yp, e1, e)
		}
		sumabscoef := 0.
		for _, c := range coef_ {
			sumabscoef += math.Abs(c)
		}
		e = e/float(len(X))/2. + self.Alpha*sumabscoef
		return e
	}
	p := Problem{}
	p.Func = squares
	p.Grad = func(grad, coef_ []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, squares, coef_, settings)

	}
	initialcoefs_ := make([]float, n_features, n_features)
	for j := 0; j < n_features; j++ {
		initialcoefs_[j] = rand.Float64()
	}
	settings := DefaultSettings()
	settings.FunctionThreshold = self.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if self.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = self.NJobs
	}

	// printer := NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &CG{}
	res, err := Local(p, initialcoefs_, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	self.Coef_ = res.X
	self._set_intercept(X_offset_, y_offset_, X_scale_)

	return self
}

func (self *Lasso) Predict(X [][]float) (y_mean []float) {
	y_mean = self.DecisionFunction(X)
	return
}

// ---
// ---
type Penalty int

const (
	penalty_L2 Penalty = iota
	penalty_L1
	penalty_elasticnet
)

type SGDRegressor struct {
	LinearModel
	base.RegressorMixin
	LearningRate, Tol, Alpha, L1_ratio float
	penalty                            Penalty
	NJobs                              int
}

func NewSGDRegressor() *SGDRegressor {
	self := &SGDRegressor{Tol: 1e-4, Alpha: 0.0001, L1_ratio: 0.15, NJobs: 1}
	self.LinearModel.FitIntercept = true
	self.RegressorMixin.Predicter = self
	return self
}

func (self *SGDRegressor) Fit(X0 [][]float, y0 []float) *SGDRegressor {
	var X, y, X_offset_, y_offset_, X_scale_ = preprocess_data(
		X0, y0, self.FitIntercept, self.Normalize)
	self.X_offset_ = X_offset_
	self.X_scale_ = X_scale_
	/*
		gd := base.NewGD()
		gd.LearningRate = self.LearningRate
		gd.Tol = self.Tol
		gd.Fit(X, y)
		//fmt.Printf("SGD X_offset_ %g, y_offset_ %g, X_scale_ %g", X_offset_, y_offset_, X_scale_)
		xtest := []float{7, 8, 9}
		floats.Sub(xtest, X_offset_)
		floats.Div(xtest, X_scale_)
		//fmt.Printf("SGD predict %g\n", gd.Predict([][]float{xtest})[0]+y_offset_)
		self.Coef_ = gd.Coefs_[1:]
	*/
	// begin use gonum gradientDescent
	n_features := len(X[0])
	loss := func(coef_ []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		tmp := make([]float, n_features, n_features)
		// e will be sum of squares of errors
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(tmp, coef_, Xi))
			e += e1 * e1
			//fmt.Printf("coef_ %v yi %g yp %g e1 %g e %g\n", coef_, y[i], yp, e1, e)
		}
		// compute regularization term R
		L1 := 0.
		L2_2 := 0.
		for _, c := range coef_[1:] {
			L1 += math.Abs(c)
			L2_2 += c * c
		}
		R := self.L1_ratio*L1 + (1.-self.L1_ratio)*L2_2
		return (e + self.Alpha*R) / float(len(X))
	}
	p := Problem{}
	p.Func = loss
	p.Grad = func(grad, coef_ []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef_, settings)

	}

	initialcoefs_ := make([]float, n_features, n_features)
	for j := 0; j < n_features; j++ {
		initialcoefs_[j] = rand.Float64()
	}
	settings := DefaultSettings()
	settings.FunctionThreshold = self.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if self.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = self.NJobs
	}

	// printer := NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &GradientDescent{}
	res, err := Local(p, initialcoefs_, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	self.Coef_ = res.X

	// end use gonum gradient gradientDescent
	self._set_intercept(X_offset_, y_offset_, X_scale_)

	return self
}

func (self *SGDRegressor) Predict(X [][]float) (y_mean []float) {
	y_mean = self.DecisionFunction(X)
	return
}

// ---
func fill(n int, x float) []float {
	var y = make([]float, n, n)
	for i := range y {
		y[i] = x
	}
	return y
}

func ones(n int) []float { return fill(n, 1.) }

func log(x float) float { return math.Log(x) }

func (self *LinearModel) _set_intercept(X_offset []float, y_offset float, X_scale []float) {
	// """Set the intercept_
	// """
	if self.FitIntercept {
		//self.Coef_ = self.Coef_ / X_scale
		floats.Div(self.Coef_, X_scale)

		//self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
		sumxoffsetcoef := 0.
		for j, Xoffj := range X_offset {
			sumxoffsetcoef += Xoffj * self.Coef_[j]
		}
		self.Intercept_ = y_offset - sumxoffsetcoef
	} else {
		self.Intercept_ = 0.
	}
}

func (self *LinearModel) DecisionFunction(X [][]float) (y []float) {
	y = make([]float, len(X))
	for i, Xi := range X {
		y[i] = self.Intercept_
		for j, c := range self.Coef_ {
			y[i] += c * Xi[j]
		}
	}

	return
}

func preprocess_data(X [][]float, y []float, fit_intercept bool, normalize bool) (
	Xout [][]float, yout []float, X_offset_ []float, y_offset_ float, X_scale_ []float) {
	var n_samples, n_features = len(X), len(X[0])
	Xout = make([][]float, n_samples, n_samples)
	yout = make([]float, n_samples)
	X_offset_ = make([]float, n_features)
	X_scale_ = make([]float, n_features)
	y_offset_ = 0.
	if fit_intercept {
		for _, Xi := range X {
			floats.Add(X_offset_, Xi)
		}
		floats.Scale(1./float(n_samples), X_offset_)

		y_offset_ = floats.Sum(y) / float(n_samples)

		if normalize {

			var X_var = make([]float, n_features)
			for _, Xi := range X {
				var t []float = make([]float, n_features)
				floats.Add(t, Xi)
				floats.Sub(t, X_offset_)
				floats.Mul(t, t)
				floats.Add(X_var, t)
			}
			floats.Scale(1./float(n_samples), X_var)
			for i, Xi := range X_var {
				X_scale_[i] = math.Sqrt(Xi)
			}
		} else {
			// no normalize
			for i := range X_scale_ {
				X_scale_[i] = 1.
			}
		}
		for i, Xi := range X {
			Xout[i] = make([]float, n_features, n_features)
			floats.Add(Xout[i], Xi)
			floats.Sub(Xout[i], X_offset_)
			floats.Div(Xout[i], X_scale_)
		}
		floats.Add(yout, y)
		floats.AddConst(-y_offset_, yout)

	} else {
		// no fit intercept
		copy(Xout, X)
		copy(yout, y)
		for i := range X_scale_ {
			X_scale_[i] = 1.
		}
	}
	return

}
