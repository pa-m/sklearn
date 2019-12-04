package naivebayes

import (
	"fmt"
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"runtime"
	"sort"
)

var _ base.Fiter = &GaussianNB{}
var _ base.Predicter = &GaussianNB{}

// BaseNB is the abstract base class for naive Bayes estimators
type BaseNB struct {
	nOutputs           int
	jointLogLikelihood func(jll, xrow []float64)
	Classes            []float64
}

// GetNOutputs ...
func (m *BaseNB) GetNOutputs() int {
	return m.nOutputs
}

// Predict perform classification on an array of test vectors X
func (m *BaseNB) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	Ypred := base.ToDense(Y)
	if Y == mat.Mutable(nil) {
		NSamples, _ := X.Dims()
		Ypred = mat.NewDense(NSamples, m.GetNOutputs(), nil)
	}
	//jll = self._joint_log_likelihood(X)
	//return self.classes_[np.argmax(jll, axis=1)]
	NSamples, NFeatures := X.Dims()
	row := make([]float64, NFeatures)
	jll := make([]float64, len(m.Classes))
	for i := 0; i < NSamples; i++ {
		mat.Row(row, i, X)
		m.jointLogLikelihood(jll, row)
		index := floats.MaxIdx(jll)
		Ypred.Set(i, 0, m.Classes[index])
	}
	return base.FromDense(Y, Ypred)
}

// PredictLogProbas return log-probability estimates.
func (m *BaseNB) PredictLogProbas(Xmatrix mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymutable)
	Xraw := X.RawMatrix()
	if Y.IsEmpty() {
		fanOut := len(m.Classes)
		Y = mat.NewDense(Xraw.Rows, fanOut, nil)
	}

	base.Parallelize(runtime.GOMAXPROCS(0), Xraw.Rows, func(th, start, end int) {
		for i := start; i < end; i++ {
			jll := Y.RawRowView(i)
			m.jointLogLikelihood(jll, Xraw.Data[i*Xraw.Stride:i*Xraw.Stride+Xraw.Cols])
			//log_prob_x = logsumexp(jll, axis=1)
			logProbX := floats.LogSumExp(jll)
			//return jll - np.atleast_2d(log_prob_x).T
			floats.AddConst(-logProbX, jll)

		}
	})

	return base.FromDense(Ymutable, Y)
}

// PredictProbas return log-probability estimates.
func (m *BaseNB) PredictProbas(Xmatrix mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymutable)
	if Y.IsEmpty() {
		fanOut := len(m.Classes)
		Y = mat.NewDense(X.RawMatrix().Rows, fanOut, nil)
	}
	m.PredictLogProbas(X, Y)
	Y.Apply(func(_, _ int, v float64) float64 { return math.Exp(v) }, Y)
	return base.FromDense(Ymutable, Y)
}

// IsClassifier ...
func (m *GaussianNB) IsClassifier() bool {
	return true
}

//GaussianNB is Gaussian Naive Bayes (GaussianNB)
//Can perform online updates to model parameters via `partial_fit` method.
//For details on algorithm used to update feature means and variance online,
//see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:
//http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
//Read more in the :ref:`User Guide <gaussian_naive_bayes>`.
type GaussianNB struct {
	Priors       []float64
	VarSmoothing float64

	ClassPrior   []float64
	ClassCount   []float64
	Theta        *mat.Dense
	Sigma        *mat.Dense
	Epsilon      float64
	SampleWeight []float64
	BaseNB
}

// NewGaussianNB ... see GaussianNB
func NewGaussianNB(priors []float64, varSmoothing float64) *GaussianNB {
	return &GaussianNB{
		Priors:       priors,
		VarSmoothing: varSmoothing,
		Epsilon:      1e-9,
		BaseNB:       BaseNB{nOutputs: 1},
	}
}

// Score returns AccuracyScore
func (m *BaseNB) Score(X, Y mat.Matrix) float64 {
	Ypred := m.Predict(X, nil)
	return metrics.AccuracyScore(Y, Ypred, true, nil)
}

// PredicterClone return a cloned GaussianNB as base.Predicter
func (m *GaussianNB) PredicterClone() base.Predicter {
	clone := *m
	return &clone
}

//Fit fit Gaussian Naive Bayes according to X, y
func (m *GaussianNB) Fit(X, Y mat.Matrix) base.Fiter {
	var Yv = colAsVector(Y, 0)
	m.PartialFit(X, Y, npUnique(Yv), true, m.SampleWeight)
	return m
}

//PartialFit fit Gaussian Naive Bayes according to X, y
func (m *GaussianNB) PartialFit(X, Y mat.Matrix, classes []float64, refit bool, sampleWeight []float64) base.Fiter {
	yr, yc := Y.Dims()
	if yc != 1 {
		panic("GaussianNB fit: expected Y to have 1 column")
	}
	//# If the ratio of data variance between dimensions is too small, it
	//# will cause numerical errors. To address this, we artificially
	//# boost the variance by epsilon, a small fraction of the standard
	//# deviation of the largest dimension.
	//	self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
	_, xc := X.Dims()

	_, varX, _ := meanvar(matfiltered{Matrix: X, filter: func(int) bool { return true }}, sampleWeight)
	m.Epsilon = m.VarSmoothing * floats.Max(varX)
	if refit {
		m.Classes = nil
	}
	firstCall := m.Theta == nil
	if firstCall {
		m.Classes = classes
		nFeatures := xc
		nClasses := len(m.Classes)
		m.Theta = mat.NewDense(nClasses, nFeatures, nil)
		m.Sigma = mat.NewDense(nClasses, nFeatures, nil)
		m.ClassCount = make([]float64, nClasses)
		//# Initialise the class prior
		//# Take into account the priors
		if m.Priors != nil {
			//priors = np.asarray(self.priors)
			priors := m.Priors
			//# Check that the provide prior match the number of classes
			if len(priors) != nClasses {
				panic("Number of priors must match number of classes.")
			}
			//# Check that the sum is 1
			priorsSum := floats.Sum(priors)
			if math.Abs(priorsSum-1.) > 1e-6 {
				panic("The sum of the priors should be 1.")
			}
			//# Check that the prior are non-negative
			priorsMin := floats.Min(priors)
			if priorsMin < 0 {
				panic("Priors must be non-negative.")
			}
			m.ClassPrior = make([]float64, len(priors))
			copy(m.ClassPrior, priors)
		} else {
			m.ClassPrior = make([]float64, len(m.Classes))
		}

	} else {
		width := func(X mat.Matrix) int { _, c := X.Dims(); return c }
		wx, wt := width(X), width(m.Theta)
		if wx != wt {
			panic(fmt.Errorf("Number of features %d does not match previous data %d", wx, wt))
		}
		//# Put epsilon back in each time
		//self.sigma_[:, :] -= self.epsilon_
		m.Sigma.Apply(func(_, _ int, v float64) float64 { return v - m.Epsilon }, m.Sigma)
	}
	classes = m.Classes
	uniqueY := npUnique(colAsVector(Y, 0))
	var uniqueYnotinClasses []float64
	classmap := map[float64]int{}
	for index, yval := range classes {
		classmap[yval] = index
	}
	for _, yval := range uniqueY {
		if _, ok := classmap[yval]; !ok {
			uniqueYnotinClasses = append(uniqueYnotinClasses, yval)
		}
	}
	if len(uniqueYnotinClasses) > 0 {
		panic(fmt.Errorf("The target labels %g in y do not exist in the initial classes %g", uniqueYnotinClasses, classes))
	}
	base.Parallelize(runtime.GOMAXPROCS(0), len(uniqueY), func(th, start, end int) {
		for i := start; i < end; i++ {
			yi := uniqueY[i]
			i := classmap[yi]
			filter := func(j int) bool { return Y.At(j, 0) == yi }
			Xi := matfiltered{Matrix: X, filter: filter}
			var swi floatsfiltered
			swi = floatsfiltered{sampleWeight, filter, yr}
			setTotalMu := func(c int, v float64) { m.Theta.Set(i, c, v) }
			setTotalVar := func(c int, v float64) { m.Sigma.Set(i, c, v) }
			setClassCount := func(v float64) { m.ClassCount[i] = v }
			m.updateMeanVariance(m.ClassCount[i], m.Theta.RawRowView(i), m.Sigma.RawRowView(i), Xi, swi, setTotalMu, setTotalVar, setClassCount)
		}
	})
	m.Sigma.Apply(func(_, _ int, v float64) float64 { return v + m.Epsilon }, m.Sigma)
	// # Update if only no priors is provided
	//
	if m.Priors == nil {
		//# Empirical prior, with sample_weight taken into account
		//self.class_prior_ = self.class_count_ / self.class_count_.sum()
		floats.ScaleTo(m.ClassPrior, 1/floats.Sum(m.ClassCount), m.ClassCount)
	}
	m.jointLogLikelihood = func(jll, xrow []float64) {
		/*
		   joint_log_likelihood = []
		      for i in range(np.size(self.classes_)):
		          jointi = np.log(self.class_prior_[i])
		          n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
		          n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
		                               (self.sigma_[i, :]), 1)
		          joint_log_likelihood.append(jointi + n_ij)


		*/
		for i := range m.Classes {
			jointi := math.Log(m.ClassPrior[i])
			nij := 0.
			sigmai := m.Sigma.RawRowView(i)
			thetai := m.Theta.RawRowView(i)
			for j := 0; j < len(xrow); j++ {
				nij -= .5 * math.Log(2*math.Pi*sigmai[j])
				xd := xrow[j] - thetai[j]
				nij -= .5 * xd * xd / sigmai[j]
			}
			jll[i] = jointi + nij
		}
	}
	return m
}

type matfiltered struct {
	mat.Matrix
	filter func(row int) bool
}

func meanvar(X matfiltered, sw []float64) (meanX, varX []float64, sumw float64) {
	xr, xc := X.Dims()

	meanX = make([]float64, xc)
	varX = make([]float64, xc)

	base.Parallelize(runtime.GOMAXPROCS(0), xc, func(th, start, end int) {
		vcol := make([]float64, xr)
		for c := start; c < end; c++ {
			mat.Col(vcol, c, X)
			xrf := 0.
			for i, val := range vcol {
				if X.filter(i) {
					var w = 1.
					if sw != nil {
						w = sw[i]
					}
					meanX[c] += val * w
					xrf += w
				}
			}
			meanX[c] /= float64(xrf)
			for i := range vcol {
				if X.filter(i) {
					var w = 1.
					if sw != nil {
						w = sw[i]
					}
					d := vcol[i] - meanX[c]
					varX[c] += d * d * w
				}
			}
			varX[c] /= float64(xrf)
			if c == 0 {
				sumw = xrf
			}

		}
	})
	return meanX, varX, sumw
}

func colAsVector(Y mat.Matrix, index int) mat.Vector {
	var Yv mat.Vector
	yr, _ := Y.Dims()
	cv, ok := Y.(mat.ColViewer)
	if ok {
		Yv = cv.ColView(index)
	} else {
		Yvd := mat.NewVecDense(yr, nil)
		mat.Col(Yvd.RawVector().Data, 0, Y)
		Yv = Yvd
	}
	return Yv
}
func npUnique(v mat.Vector) []float64 {
	classmap := map[float64]int{}
	for i := 0; i < v.Len(); i++ {
		val := v.AtVec(i)
		classmap[val] = i
	}
	classlist := make([]float64, 0, len(classmap))
	for cl := range classmap {
		classlist = append(classlist, cl)
	}
	sort.Float64s(classlist)
	return classlist
}

type floatsfiltered struct {
	sw      []float64
	filter  func(int) bool
	samples int
}

func (m *GaussianNB) updateMeanVariance(nPast float64, mu, va []float64, X matfiltered, sw floatsfiltered,
	setTotalMu, setTotalVar func(c int, v float64),
	setClassCount func(float64)) {
	_, xc := X.Dims()
	xr := 0
	for j := 0; j < sw.samples; j++ {
		if sw.filter(j) {
			xr++
		}
	}
	if xr == 0 {
		return
	}
	newMu, newVa, nNew := meanvar(X, sw.sw)
	nTotal := nPast + nNew
	for c := 0; c < xc; c++ {
		setTotalMu(c, (nPast*mu[c]+nNew*newMu[c])/nTotal)
		dmu := mu[c] - newMu[c]
		setTotalVar(c, (nPast*va[c]+nNew*newVa[c]+nNew*nPast/nTotal*dmu*dmu)/nTotal)
	}
	setClassCount(nTotal)
	return
}
