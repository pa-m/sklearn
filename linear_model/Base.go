package linear_model

import (
	_ "fmt"
	"github.com/gonum/floats"
	_ "github.com/gonum/stat"
	"github.com/pa-m/sklearn/base"
	_ "gonum.org/v1/gonum/mat"
	"math"
)

type float = float64

type LinearModel struct {
	X_offset_, X_scale_             []float
	Coef_                           []float
	Intercept_                      float
	FitIntercept, Normalize, Copy_X bool
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
// copy_X : boolean, optional, default True
//     If True, X will be copied; else, it may be overwritten.
// n_jobs : int, optional, default 1
//     The number of jobs to use for the computation.
//     If -1 all CPUs are used. This will only provide speedup for
//     n_targets > 1 and sufficient large problems.
// Attributes
// ----------
// coef_ : array, shape (n_features, ) or (n_targets, n_features)
//     Estimated coefficients for the linear regression problem.
//     If multiple targets are passed during the fit (y 2D), this
//     is a 2D array of shape (n_targets, n_features), while if only
//     one target is passed, this is a 1D array of length n_features.
// intercept_ : array
//     Independent term in the linear model.
// Notes
// -----
// From the implementation point of view, this is just plain Ordinary
// Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
// """

type LinearRegression struct {
	base.RegressorMixin
}

func (self *LinearRegression) Fit(X0 [][]float, y0 []float) *LinearRegression {
	return self
}
func (self *LinearRegression) Predict(X [][]float) (y_mean []float) {
	y_mean = make([]float, len(X), len(X))
	return
}

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

func preprocess_data(X [][]float, y []float, fit_intercept bool, normalize bool, copy bool) (
	Xout [][]float, yout []float, X_offset_ []float, y_offset_ float, X_scale_ []float) {
	var n_samples, n_features = len(X), len(X[0])
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
		Xout = make([][]float, n_samples, n_samples)
		for i, Xi := range X {
			Xout[i] = make([]float, n_features, n_features)
			floats.Add(Xout[i], Xi)
			floats.Sub(Xout[i], X_offset_)
			floats.Div(Xout[i], X_scale_)
		}
		yout = make([]float, n_samples)
		floats.Add(yout, y)
		floats.AddConst(-y_offset_, yout)

	} else {
		// no fit intercept
		for i := range X_scale_ {
			X_scale_[i] = 1.
		}
	}
	return

}
