package linear_model

import (
	"fmt"
	"github.com/gonum/floats"
	"github.com/gonum/stat"
	"gonum.org/v1/gonum/mat"
	"math"
)

type LinearModel struct {
	X_offset_, X_scale_ []float
	Coef_               []float
	Intercept_          float
	FitIntercept        bool
}
type RegressorMixin struct{}

type float = float64

type BayesianRidge struct {
	LinearModel
	RegressorMixin
	N_iter                                    int
	Tol, Alpha_1, Alpha_2, Lambda_1, Lambda_2 float
	ComputeScore, Normalize, Copy_X, Verbose  bool
	Alpha_, Lambda_                           float
	Sigma_                                    []float
	Scores_                                   []float
}

func NewBayesianRidge() *BayesianRidge {
	return &BayesianRidge{LinearModel: LinearModel{FitIntercept: true}, RegressorMixin: RegressorMixin{}, N_iter: 300, Tol: 1e-3, Alpha_1: 1e-6, Alpha_2: 1e-6,
		Lambda_1: 1e-6, Lambda_2: 1e-6, ComputeScore: false, Normalize: false, Copy_X: true, Verbose: false,
	}
}

// Fit the model
//         Parameters
//         ----------
//         X : numpy array of shape [n_samples,n_features]
//             Training data
//         y : numpy array of shape [n_samples]
//             Target values. Will be cast to X's dtype if necessary

func (self *BayesianRidge) Fit(X0 [][]float, y0 []float) *BayesianRidge {
	var n_samples, n_features = len(X0), len(X0[0])
	var X, y, X_offset_, y_offset_, X_scale_ = preprocess_data(
		X0, y0, self.FitIntercept, self.Normalize, self.Copy_X)
	self.X_offset_ = X_offset_
	self.X_scale_ = X_scale_
	alpha_ := 1. / stat.Variance(y, ones(n_samples))
	lambda_ := 1.
	verbose := self.Verbose
	lambda_1 := self.Lambda_1
	lambda_2 := self.Lambda_2
	alpha_1 := self.Alpha_1
	alpha_2 := self.Alpha_2
	self.Scores_ = make([]float, 0)
	var coef_old_ *mat.Dense = mat.NewDense(n_features, 1, nil)
	var logdet_sigma_ float
	XT_y := mat.NewDense(n_features, 1, nil)
	var coef_, sigma_ *mat.Dense
	for j := 0; j < n_features; j++ {
		var XT_yj = 0.
		for i := range X {
			XT_yj += X[i][j] * y[i]
		}
		XT_y.Set(j, 0, XT_yj)
	}
	Xm := mat.NewDense(n_samples, n_features, nil)
	for i, Xi := range X {
		Xm.SetRow(i, Xi)
	}
	var svd mat.SVD
	if !svd.Factorize(Xm, mat.SVDThin) {
		panic("svd failed")
	}
	U, S, VhT := svd.UTo(nil), svd.Values(nil), svd.VTo(nil)

	U = U
	eigen_vals_ := make([]float, len(S), len(S))
	for j, Sj := range S {
		eigen_vals_[j] = Sj * Sj
	}

	calcsigma := func(VhT *mat.Dense, eigen_vals_ []float, lambda_, alpha_ float) *mat.Dense {
		// compute np.dot(Vh.T,Vh / (eigen_vals_ +lambda_ / alpha_)[:, np.newaxis])
		coef_ := mat.NewDense(n_features, n_features, nil)
		right := mat.NewDense(n_features, n_features, nil)
		for i := 0; i < n_features; i++ {
			rightrow := make([]float, n_features, n_features)
			mat.Col(rightrow, i, VhT)
			floats.Scale(1./(eigen_vals_[i]+lambda_/alpha_), rightrow)
			right.SetRow(i, rightrow)
		}
		coef_.Mul(VhT, right)
		return coef_
	}
	dimstr := func(m *mat.Dense) string { r, c := m.Dims(); return fmt.Sprint("(%d x %d)", r, c) }
	dimstr = dimstr
	// # Convergence loop of the bayesian ridge regression
	for iter_ := 0; iter_ < self.N_iter; iter_++ {
		// # Compute mu and sigma
		// # sigma_ = lambda_ / alpha_ * np.eye(n_features) + np.dot(X.T, X)
		// # coef_ = sigma_^-1 * XT * y
		if n_samples > n_features {
			// coef_ = np.dot(Vh.T,Vh / (eigen_vals_ +lambda_ / alpha_)[:, np.newaxis])
			coeftmp_ := calcsigma(VhT, eigen_vals_, lambda_, alpha_)
			//coef_ = np.dot(coef_, XT_y)
			coef_ = mat.NewDense(n_features, 1, nil)
			coef_.Mul(coeftmp_, XT_y)
			if self.ComputeScore {
				//logdet_sigma_ = - np.sum(np.log(lambda_ + alpha_ * eigen_vals_))
				logdet_sigma_ = 0.
				for _, evi := range eigen_vals_ {
					logdet_sigma_ += math.Log(lambda_ + alpha_*evi)
				}
			}
		} else { // n_samples<=n_features
			panic("unimplemented n_samples<=n_features")
		}
		// # Preserve the alpha and lambda values that were used to
		// # calculate the final coefficients
		self.Alpha_ = alpha_
		self.Lambda_ = lambda_
		// # Update alpha and lambda
		// rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)
		rmse_ := 0.
		for i, Xi := range X {
			sumxc := 0.
			for j, Xij := range Xi {
				sumxc += Xij * coef_.At(j, 0)
			}
			rmse_ += math.Pow(y[i]-sumxc, 2.)
		}
		// gamma_ = (np.sum((alpha_ * eigen_vals_) /(lambda_ + alpha_*eigen_vals_)))
		gamma_ := 0.
		{
			var left, right []float
			copy(left, eigen_vals_)

			floats.Scale(alpha_, left)
			copy(right, left)
			floats.AddConst(lambda_, right)
			floats.Div(left, right)
			gamma_ = floats.Sum(left)
		}
		//lambda_ = ((gamma_ + 2*lambda_1) /(np.sum(coef_**2) + 2*lambda_2))
		lambda_ = 0.
		sumcoef2 := 0.
		{
			var coef2 []float = make([]float, n_features, n_features)
			mat.Col(coef2, 0, coef_)
			floats.Mul(coef2, coef2)
			sumcoef2 = floats.Sum(coef2)
			lambda_ = (gamma_ + 2*lambda_1) / (sumcoef2 + 2*lambda_2)

		}
		alpha_ = ((float(n_samples) - gamma_ + 2*alpha_1) / (rmse_ + 2*alpha_2))
		// # Compute the objective function
		if self.ComputeScore {
			s := lambda_1*log(lambda_) - lambda_2*lambda_
			s += alpha_1*log(alpha_) - alpha_2*alpha_
			s += 0.5 * (float(n_features)*log(lambda_) +
				float(n_samples)*log(alpha_) -
				alpha_*rmse_ -
				(lambda_ * sumcoef2) -
				logdet_sigma_ -
				float(n_samples)*log(2*math.Pi))
			self.Scores_ = append(self.Scores_, s)
		}
		// # Check for convergence
		if iter_ > 0 {
			sumabsdiff := 0.
			for j := 0; j < n_features; j++ {
				sumabsdiff += math.Abs(coef_old_.At(j, 0) - coef_.At(j, 0))
			}
			if sumabsdiff < self.Tol {
				if verbose {
					fmt.Println("Convergence after ", iter_, " iterations")
				}
				break
			}
		}

		coef_old_.Copy(coef_)

	}
	self.Coef_ = make([]float, n_features)
	mat.Col(self.Coef_, 0, coef_)
	//sigma_ = np.dot(Vh.T,Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis])
	//self.sigma_ = (1. / alpha_) * sigma_
	sigma_ = calcsigma(VhT, eigen_vals_, lambda_, alpha_)
	mat.Col(self.Sigma_, 0, sigma_)
	floats.Scale(1./alpha_, self.Sigma_)

	self._set_intercept(X_offset_, y_offset_, X_scale_)
	return self
	//https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/linear_model/bayes.py#L23
}

// """Predict using the linear model.
// In addition to the mean of the predictive distribution, also its
// standard deviation can be returned.
// Parameters
// ----------
// X : {array-like, sparse matrix}, shape = (n_samples, n_features)
//     Samples.
// Returns
// -------
// y_mean : array, shape = (n_samples,)
//     Mean of predictive distribution of query points.
// """
func (self *BayesianRidge) Predict(X [][]float) (y_mean []float) {
	y_mean = self.DecisionFunction(X)
	return
}
func (self *BayesianRidge) Predict2(X0 [][]float) (y_mean, y_std []float) {
	y_mean = self.DecisionFunction(X0)
	var X [][]float
	copy(X, X0)
	if self.Normalize {
		for i := range X {
			floats.Sub(X[i], self.X_offset_)
			floats.Div(X[i], self.X_scale_)
		}
	}
	//sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
	sigmas_squared_data := 0.
	y_std = make([]float, len(X0), len(X0))
	for i, Xi := range X {
		for j, s := range self.Sigma_ {
			sigmas_squared_data = Xi[j] * s * Xi[j]
		}
		y_std[i] = math.Sqrt(sigmas_squared_data + 1./self.Alpha_)
	}
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
