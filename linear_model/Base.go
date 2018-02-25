package linearModel

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"

	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/optimize"
)

type float = float64

// LinearModel is a base struct for some predicters
type LinearModel struct {
	XOffset, XScale         []float
	Coef                    []float
	Intercept               float
	FitIntercept, Normalize bool
	Method                  optimize.Method
}

// LinearModel2 is a base struct for some predicters
type LinearModel2 struct {
	XOffset, XScale         []float
	Coef                    [][]float
	Intercept               []float
	FitIntercept, Normalize bool
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
	base.RegressorMixin
	Tol   float
	NJobs int
}

// NewLinearRegression create a *LinearRegression with defaults
func NewLinearRegression() *LinearRegression {
	regr := &LinearRegression{Tol: 1e-6, NJobs: 1}
	regr.LinearModel.FitIntercept = true
	regr.RegressorMixin.Predicter = regr
	return regr
}

// Fit fits Coef for a LinearRegression
func (regr *LinearRegression) Fit(X0 [][]float, y0 []float) *LinearRegression {
	var nFeatures = len(X0[0])
	var X, y, XOffset, yOffset, XScale = preprocessData(
		X0, y0, regr.FitIntercept, regr.Normalize)
	regr.XOffset = XOffset
	regr.XScale = XScale
	loss := func(coef []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, nFeatures, nFeatures)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef, Xi))
			e += e1 * e1
			//fmt.Printf("coef %v yi %g yp %g e1 %g e %g\n", coef, y[i], yp, e1, e)
		}
		return e
	}
	p := optimize.Problem{}
	p.Func = loss
	p.Grad = func(grad, coef []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef, settings)

	}
	initialcoefs := make([]float, nFeatures, nFeatures)
	for j := 0; j < nFeatures; j++ {
		initialcoefs[j] = rand.Float64()
	}
	settings := optimize.DefaultSettings()
	settings.FunctionThreshold = regr.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if regr.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = regr.NJobs
	}

	// printer := optimize.NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer
	method := regr.Method
	if method == nil {
		// the fastest method seems to be CG when normalized and BFGS when non-normalized
		if regr.Normalize {
			method = &optimize.CG{}
		} else {
			method = &optimize.BFGS{}
		}
	}
	res, err := optimize.Local(p, initialcoefs, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	regr.Coef = res.X
	regr.SetIntercept(XOffset, yOffset, XScale)

	return regr
}

// Predict predicts y for X using Coef
func (regr *LinearRegression) Predict(X [][]float) (yMean []float) {
	yMean = regr.DecisionFunction(X)
	return
}

// Ridge regression base struct
type Ridge struct {
	LinearModel
	base.RegressorMixin
	Alpha, Tol float
	NJobs      int
}

// NewRidge creates a *Ridge with defaults
func NewRidge() *Ridge {
	regr := &Ridge{Alpha: 1., Tol: 1e-3, NJobs: 1}
	regr.LinearModel.FitIntercept = true
	regr.RegressorMixin.Predicter = regr
	return regr
}

// Fit lears Coef for Ridge
func (regr *Ridge) Fit(X0 [][]float, y0 []float) *Ridge {
	if regr.Normalize {
		fmt.Fprintf(os.Stderr, "Ridge don't work with Normalize. Reverting Normalize to False\n")
		regr.Normalize = false
	}
	var nFeatures = len(X0[0])
	var X, y, XOffset, yOffset, XScale = preprocessData(
		X0, y0, regr.FitIntercept, regr.Normalize)
	regr.XOffset = XOffset
	regr.XScale = XScale
	loss := func(coef []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, nFeatures, nFeatures)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef, Xi))
			e += e1 * e1
			//fmt.Printf("coef %v yi %g yp %g e1 %g e %g\n", coef, y[i], yp, e1, e)
		}
		coef2 := make([]float, nFeatures, nFeatures)
		floats.MulTo(coef2, coef, coef)
		e = e/float(len(X)) + regr.Alpha*floats.Sum(coef2)/float(len(coef))
		return e
	}
	p := optimize.Problem{}
	p.Func = loss
	p.Grad = func(grad, coef []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef, settings)

	}
	initialcoefs := make([]float, nFeatures, nFeatures)
	for j := 0; j < nFeatures; j++ {
		initialcoefs[j] = rand.Float64()
	}
	settings := optimize.DefaultSettings()
	settings.FunctionThreshold = regr.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if regr.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = regr.NJobs
	}

	// printer := optimize.NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &optimize.CG{}
	res, err := optimize.Local(p, initialcoefs, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	regr.Coef = res.X
	regr.SetIntercept(XOffset, yOffset, XScale)

	return regr
}

// Predict predicts y for X using Coef
func (regr *Ridge) Predict(X [][]float) (yMean []float) {
	yMean = regr.DecisionFunction(X)
	return
}

// Lasso regression base struct
type Lasso struct {
	LinearModel
	base.RegressorMixin
	Alpha, Tol float
	NJobs      int
}

//NewLasso creates a *Lasso with defaults
func NewLasso() *Lasso {
	regr := &Lasso{Alpha: 1., Tol: 1e-4, NJobs: 1}
	regr.LinearModel.FitIntercept = true
	regr.RegressorMixin.Predicter = regr
	return regr
}

// Fit learns Coef for Lasso
func (regr *Lasso) Fit(X0 [][]float, y0 []float) *Lasso {
	if regr.Normalize {
		fmt.Fprintf(os.Stderr, "Lasso don't work with Normalize. Reverting Normalize to False\n")
		regr.Normalize = false
	}
	var nFeatures = len(X0[0])
	var X, y, XOffset, yOffset, XScale = preprocessData(
		X0, y0, regr.FitIntercept, regr.Normalize)
	regr.XOffset = XOffset
	regr.XScale = XScale
	squares := func(coef []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		coefMulXi := make([]float, nFeatures, nFeatures)
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(coefMulXi, coef, Xi))
			e += e1 * e1
			//fmt.Printf("coef %v yi %g yp %g e1 %g e %g\n", coef, y[i], yp, e1, e)
		}
		sumabscoef := 0.
		for _, c := range coef {
			sumabscoef += math.Abs(c)
		}
		e = e/float(len(X))/2. + regr.Alpha*sumabscoef
		return e
	}
	p := optimize.Problem{}
	p.Func = squares
	p.Grad = func(grad, coef []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, squares, coef, settings)

	}
	initialcoefs := make([]float, nFeatures, nFeatures)
	for j := 0; j < nFeatures; j++ {
		initialcoefs[j] = rand.Float64()
	}
	settings := optimize.DefaultSettings()
	settings.FunctionThreshold = regr.Tol
	settings.GradientThreshold = 1.e-12
	/*  settings.FunctionConverge.Iterations = 1000
	 */
	settings.FunctionConverge = nil
	if regr.NJobs <= 0 {
		settings.Concurrent = runtime.NumCPU()
	} else {
		settings.Concurrent = regr.NJobs
	}

	// printer := optimize.NewPrinter()
	// printer.HeadingInterval = 1
	// settings.Recorder = printer

	method := &optimize.CG{}
	res, err := optimize.Local(p, initialcoefs, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	regr.Coef = res.X
	regr.SetIntercept(XOffset, yOffset, XScale)

	return regr
}

// Predict predicts y from X using Coef
func (regr *Lasso) Predict(X [][]float) (yMean []float) {
	yMean = regr.DecisionFunction(X)
	return
}

// SGDRegressor base struct
type SGDRegressor struct {
	LinearModel
	base.RegressorMixin
	LearningRate, Tol, Alpha, L1Ratio float
	NJobs                             int
}

// NewSGDRegressor creates a *SGDRegressor with defaults
func NewSGDRegressor() *SGDRegressor {
	regr := &SGDRegressor{Tol: 1e-4, Alpha: 0.0001, L1Ratio: 0.15, NJobs: 1}
	regr.LinearModel.FitIntercept = true
	regr.RegressorMixin.Predicter = regr
	return regr
}

// Fit learns Coef
func (regr *SGDRegressor) Fit(X0 [][]float, y0 []float) *SGDRegressor {
	var X, y, XOffset, yOffset, XScale = preprocessData(
		X0, y0, regr.FitIntercept, regr.Normalize)
	regr.XOffset = XOffset
	regr.XScale = XScale
	/*
		gd := base.NewGD()
		gd.LearningRate = regr.LearningRate
		gd.Tol = regr.Tol
		gd.Fit(X, y)
		//fmt.Printf("SGD XOffset %g, yOffset %g, XScale %g", XOffset, yOffset, XScale)
		xtest := []float{7, 8, 9}
		floats.Sub(xtest, XOffset)
		floats.Div(xtest, XScale)
		//fmt.Printf("SGD predict %g\n", gd.Predict([][]float{xtest})[0]+yOffset)
		regr.Coef = gd.Coefs[1:]
	*/
	// begin use gonum gradientDescent
	nFeatures := len(X[0])
	loss := func(coef []float) float {
		// e = sumi { (yi -sumj cj Xij)² }
		// de/dcj =
		tmp := make([]float, nFeatures, nFeatures)
		// e will be sum of squares of errors
		e := 0.
		for i, Xi := range X {
			e1 := y[i] - floats.Sum(floats.MulTo(tmp, coef, Xi))
			e += e1 * e1
			//fmt.Printf("coef %v yi %g yp %g e1 %g e %g\n", coef, y[i], yp, e1, e)
		}
		// compute regularization term R
		L1 := 0.
		L22 := 0.
		for _, c := range coef[1:] {
			L1 += math.Abs(c)
			L22 += c * c
		}
		R := regr.L1Ratio*L1 + (1.-regr.L1Ratio)*L22
		return (e + regr.Alpha*R) / float(len(X))
	}
	p := optimize.Problem{}
	p.Func = loss
	p.Grad = func(grad, coef []float) {
		h := 1e-6

		settings := &fd.Settings{}
		settings.Concurrent = true
		settings.Step = h
		fd.Gradient(grad, loss, coef, settings)

	}

	initialcoefs := make([]float, nFeatures, nFeatures)
	for j := 0; j < nFeatures; j++ {
		initialcoefs[j] = rand.Float64()
	}
	settings := optimize.DefaultSettings()
	settings.FunctionThreshold = regr.Tol
	settings.GradientThreshold = 1.e-12
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

	method := &optimize.GradientDescent{}
	res, err := optimize.Local(p, initialcoefs, settings, method)
	//fmt.Printf("res=%s %#v\n", res.Status.String(), res)
	if err != nil && err.Error() != "linesearch: no change in location after Linesearcher step" {

		fmt.Println(err)
	}
	regr.Coef = res.X

	// end use gonum gradient gradientDescent
	regr.SetIntercept(XOffset, yOffset, XScale)

	return regr
}

// Predict predicts y from X using Coef
func (regr *SGDRegressor) Predict(X [][]float) (yMean []float) {
	yMean = regr.DecisionFunction(X)
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

//func log(x float) float { return math.Log(x) }

// SetIntercept adjusts Coefs and Intercept using preprocess data
func (regr *LinearModel) SetIntercept(XOffset []float, yOffset float, XScale []float) {
	// """Set the intercept
	// """
	if regr.FitIntercept {
		//regr.Coef = regr.Coef / XScale
		floats.Div(regr.Coef, XScale)

		//regr.intercept = yOffset - np.dot(XOffset, regr.coef.T)
		sumxoffsetcoef := 0.
		for j, Xoffj := range XOffset {
			sumxoffsetcoef += Xoffj * regr.Coef[j]
		}
		regr.Intercept = yOffset - sumxoffsetcoef
	} else {
		regr.Intercept = 0.
	}
}

// DecisionFunction returns y from X and Coef (for preprocessed data)
func (regr *LinearModel) DecisionFunction(X [][]float) (y []float) {
	y = make([]float, len(X))
	for i, Xi := range X {
		y[i] = regr.Intercept
		for j, c := range regr.Coef {
			y[i] += c * Xi[j]
		}
	}

	return
}

func preprocessData(X [][]float, y []float, fitIntercept bool, normalize bool) (
	Xout [][]float, yout []float, XOffset []float, yOffset float, XScale []float) {
	var nSamples, nFeatures = len(X), len(X[0])
	Xout = make([][]float, nSamples, nSamples)
	yout = make([]float, nSamples)
	XOffset = make([]float, nFeatures)
	XScale = make([]float, nFeatures)
	yOffset = 0.
	if fitIntercept {
		for _, Xi := range X {
			floats.Add(XOffset, Xi)
		}
		floats.Scale(1./float(nSamples), XOffset)

		yOffset = floats.Sum(y) / float(nSamples)

		if normalize {

			var XVar = make([]float, nFeatures)
			for _, Xi := range X {
				var t = make([]float, nFeatures)
				floats.Add(t, Xi)
				floats.Sub(t, XOffset)
				floats.Mul(t, t)
				floats.Add(XVar, t)
			}
			floats.Scale(1./float(nSamples), XVar)
			for i, Xi := range XVar {
				XScale[i] = math.Sqrt(Xi)
			}
		} else {
			// no normalize
			for i := range XScale {
				XScale[i] = 1.
			}
		}
		for i, Xi := range X {
			Xout[i] = make([]float, nFeatures, nFeatures)
			floats.Add(Xout[i], Xi)
			floats.Sub(Xout[i], XOffset)
			floats.Div(Xout[i], XScale)
		}
		floats.Add(yout, y)
		floats.AddConst(-yOffset, yout)

	} else {
		// no fit intercept
		copy(Xout, X)
		copy(yout, y)
		for i := range XScale {
			XScale[i] = 1.
		}
	}
	return

}

func unused(...interface{}) {}
