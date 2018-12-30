package linearmodel

import (
	"fmt"
	"math"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// BayesianRidge regression struct
type BayesianRidge struct {
	LinearModel
	NIter                                 int
	Tol, Alpha1, Alpha2, Lambda1, Lambda2 float
	ComputeScore, Verbose                 bool
	Alpha, Lambda                         float
	Sigma                                 *mat.Dense
	Scores                                []float
}

// NewBayesianRidge creates a *BayesianRidge with defaults
func NewBayesianRidge() *BayesianRidge {
	regr := &BayesianRidge{LinearModel: LinearModel{FitIntercept: true, Normalize: true}, NIter: 300, Tol: 1e-3, Alpha1: 1e-6, Alpha2: 1e-6,
		Lambda1: 1e-6, Lambda2: 1e-6, ComputeScore: false, Verbose: false,
	}
	//regr.RegressorMixin1.Predicter = regr
	return regr
}

// Fit the model
//         Parameters
//         ----------
//         X : numpy array of shape [nSamples,nFeatures]
//             Training data
//         y : numpy array of shape [nSamples]
//             Target values. Will be cast to X's dtype if necessary
func (regr *BayesianRidge) Fit(X0, Y0 *mat.Dense) base.Transformer {
	var X, Y, YOffset *mat.Dense
	X, Y, regr.XOffset, YOffset, regr.XScale = PreprocessData(X0, Y0, regr.FitIntercept, regr.Normalize, nil)
	var nSamples, nFeatures = X0.Dims()
	var _, nOutputs = Y.Dims()
	//alpha_ = 1. / np.var(y)
	alpha := 0.
	Y.Apply(func(i int, j int, y float64) float64 {
		alpha += y * y
		return y
	}, Y)
	alpha = float64(nSamples*nOutputs) / alpha

	lambda := 1.
	verbose := regr.Verbose
	lambda1 := regr.Lambda1
	lambda2 := regr.Lambda2
	alpha1 := regr.Alpha1
	alpha2 := regr.Alpha2
	regr.Scores = make([]float, 0)
	coefOld := mat.NewDense(nFeatures, nOutputs, nil)
	var logdetSigma float
	XTY := mat.NewDense(nFeatures, nOutputs, nil)
	XTY.Mul(X.T(), Y)
	var coef, sigma *mat.Dense
	var svd mat.SVD
	if !svd.Factorize(X, mat.SVDThin) {
		panic("svd failed")
	}
	U, S, VhT := svd.UTo(nil), svd.Values(nil), svd.VTo(nil)
	unused(U)

	eigenVals := make([]float, len(S), len(S))
	for j, Sj := range S {
		eigenVals[j] = Sj * Sj
	}

	calcsigmaCoef := mat.NewDense(nFeatures, nFeatures, nil)
	calcsigmaRight := mat.NewDense(nFeatures, nFeatures, nil)
	calcsigma := func(VhT *mat.Dense, eigenVals []float, lambda, alpha float) *mat.Dense {
		// compute np.dot(Vh.T,Vh / (eigenVals +lambda / alpha)[:, np.newaxis])
		coef := calcsigmaCoef
		right := calcsigmaRight
		for i := 0; i < nFeatures; i++ {
			rightrow := make([]float, nFeatures, nFeatures)
			mat.Col(rightrow, i, VhT)
			floats.Scale(1./(eigenVals[i]+lambda/alpha), rightrow)
			right.SetRow(i, rightrow)
		}
		coef.Mul(VhT, right)
		return coef
	}
	// dimstr := func(m *mat.Dense) string { r, c := m.Dims(); return fmt.Sprint("(%d x %d)", r, c) }
	// dimstr = dimstr
	coef = mat.NewDense(nFeatures, nOutputs, nil)
	diff := mat.NewDense(nSamples, nOutputs, nil)

	coef2 := mat.NewDense(nFeatures, nOutputs, nil)

	// # Convergence loop of the bayesian RidgeMatMat regression
	for iter := 0; iter < regr.NIter; iter++ {
		// # Compute mu and sigma
		// # sigma = lambda / alpha * np.eye(nFeatures) + np.dot(X.T, X)
		// # coef = sigma^-1 * XT * y
		if nSamples > nFeatures {
			// coef = np.dot(Vh.T,Vh / (eigenVals +lambda / alpha)[:, np.newaxis])
			coeftmp := calcsigma(VhT, eigenVals, lambda, alpha)
			//coef = np.dot(coef, XTY)
			coef.Mul(coeftmp, XTY)
			if regr.ComputeScore {
				//logdetSigma = - np.sum(np.log(lambda + alpha * eigenVals))
				logdetSigma = 0.
				for _, evi := range eigenVals {
					logdetSigma += math.Log(lambda + alpha*evi)
				}
			}
		} else { // nSamples<=nFeatures
			panic("unimplemented nSamples<=nFeatures")
		}
		// # Preserve the alpha and lambda values that were used to
		// # calculate the final coefficients
		regr.Alpha = alpha
		regr.Lambda = lambda
		// # Update alpha and lambda
		// rmse = np.sum((y - np.dot(X, coef)) ** 2)
		diff.Mul(X, coef)
		diff.Sub(diff, Y)

		rmse := mat.Norm(diff, 2)

		// gamma = (np.sum((alpha * eigenVals) /(lambda + alpha*eigenVals)))
		gamma := 0.
		{
			var left, right []float
			copy(left, eigenVals)

			floats.Scale(alpha, left)
			copy(right, left)
			floats.AddConst(lambda, right)
			floats.Div(left, right)
			gamma = floats.Sum(left)
		}
		//lambda = ((gamma + 2*lambda1) /(np.sum(coef**2) + 2*lambda2))
		coef2.MulElem(coef, coef)
		sumcoef2 := mat.Sum(coef2)
		lambda = (gamma + 2*lambda1) / (mat.Sum(coef2) + 2*lambda2)
		//alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /(rmse_ + 2 * alpha_2))
		alpha = ((float(nSamples) - gamma + 2*alpha1) / (rmse + 2*alpha2))
		log := math.Log
		// # Compute the objective function
		if regr.ComputeScore {
			s := lambda1*log(lambda) - lambda2*lambda
			s += alpha1*log(alpha) - alpha2*alpha
			s += 0.5 * (float(nFeatures)*log(lambda) +
				float(nSamples)*log(alpha) -
				alpha*rmse -
				(lambda * sumcoef2) -
				logdetSigma -
				float(nSamples)*log(2*math.Pi))
			regr.Scores = append(regr.Scores, s)
		}
		// # Check for convergence
		if iter > 0 {
			sumabsdiff := 0.
			for j := 0; j < nFeatures; j++ {
				for o := 0; o < nOutputs; o++ {
					sumabsdiff += math.Abs(coefOld.At(j, o) - coef.At(j, o))
				}
			}
			if sumabsdiff < regr.Tol {
				if verbose {
					fmt.Println("Convergence after ", iter, " iterations")
				}
				break
			}
		}

		coefOld.Clone(coef)

	}
	//sigma = np.dot(Vh.T,Vh / (eigenVals + lambda / alpha)[:, np.newaxis])
	//regr.sigma = (1. / alpha) * sigma
	sigma = calcsigma(VhT, eigenVals, lambda, alpha)
	regr.Alpha = alpha
	regr.Sigma = mat.NewDense(nFeatures, nFeatures, nil)
	regr.Sigma.Scale(1./alpha, sigma)

	//set_intercept
	regr.Coef = coef
	regr.Intercept = mat.NewDense(1, nOutputs, nil)
	regr.setIntercept(regr.XOffset, YOffset, regr.XScale)
	return regr
}

// Predict using the linear model.
// In addition to the mean of the predictive distribution, also its
// standard deviation can be returned.
// Parameters
// ----------
// X : {array-like, sparse matrix}, shape = (nSamples, nFeatures)
//     Samples.
// Returns
// -------
// yMean : array, shape = (nSamples,)
//     Mean of predictive distribution of query points.
// """
func (regr *BayesianRidge) Predict(X, Y *mat.Dense) base.Regressor {
	// d := func(X mat.Matrix) string { r, c := X.Dims(); return fmt.Sprintf("%d,%d", r, c) }
	// fmt.Println("Predict", d(X), d(regr.Coef))
	Y.Mul(X, regr.Coef)
	Y.Apply(func(i int, j int, y float64) float64 {
		return y + regr.Intercept.At(0, j)
	}, Y)
	return regr
}

// Predict2 returns y and stddev
func (regr *BayesianRidge) Predict2(X, Y, yStd *mat.Dense) base.Regressor {
	nSamples, nFeatures := X.Dims()
	regr.Predict(X, Y)
	Xn := mat.DenseCopyOf(X)
	if regr.Normalize {
		Xn.Apply(func(i int, j int, xij float64) float64 {
			return (xij - regr.XOffset.At(0, j)) / regr.XScale.At(0, j)
		}, Xn)
	}
	//sigmasSquaredData = (np.dot(X, regr.sigma) * X).sum(axis=1)
	//y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
	sigmasSquaredData := mat.NewDense(nSamples, 1, nil)
	sigmasSquaredData.Apply(func(i int, _ int, _ float64) float64 {
		v := 0.
		for j := 0; j < nFeatures; j++ {
			x := Xn.At(i, j)
			v += x * regr.Sigma.At(0, j) * x
		}
		return v
	}, sigmasSquaredData)
	yStd = mat.NewDense(nSamples, 1, nil)
	yStd.Apply(func(i int, j int, sigmasSquaredData float64) float64 {
		return sigmasSquaredData + 1./regr.Alpha
	}, sigmasSquaredData)

	return regr
}

// FitTransform is for Pipeline
func (regr *BayesianRidge) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	r, c := Y.Dims()
	Xout, Yout = X, mat.NewDense(r, c, nil)
	regr.Fit(X, Y)
	regr.Predict(X, Yout)
	return
}

// Transform is for Pipeline
func (regr *BayesianRidge) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	r, c := Y.Dims()
	Xout, Yout = X, mat.NewDense(r, c, nil)
	regr.Predict(X, Yout)
	return
}
