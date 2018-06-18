package svm

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
)

// links:
// https://www.quora.com/What-does-support-vector-machine-SVM-mean-in-laymans-terms
// https://github.com/ewalker544/libsvm-go
// https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
// v andrew NG "machine learning" ex6

// Model for SVM
type Model struct {
	X              *mat.Dense
	Y              []float64
	KernelFunction func(X1, X2 []float64) float64
	B              float64
	Alphas         []float64
	W              []float64
}

// %svmTrain Trains an SVM classifier using a simplified version of the SMO
// %algorithm.
// %   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
// %   SVM classifier and returns trained model. X is the matrix of training
// %   examples.  Each row is a training example, and the jth column holds the
// %   jth feature.  Y is a column matrix containing 1 for positive examples
// %   and 0 for negative examples.  C is the standard SVM regularization
// %   parameter.  tol is a tolerance value used for determining equality of
// %   floating point numbers. max_passes controls the number of iterations
// %   over the dataset (without changes to alpha) before the algorithm quits.
// %
// % Note: This is a simplified version of the SMO algorithm for training
// %       SVMs. In practice, if you want to train an SVM classifier, we
// %       recommend using an optimized package such as:
// %
// %           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
// %           SVMLight (http://svmlight.joachims.org/)
func svmTrain(X *mat.Dense, Y []float64, C float64, KernelFunction func(X1, X2 []float64) float64, Tol float64, MaxPasses, CacheSize int) *Model {
	m, n := X.Dims()
	alphas := make([]float64, m, m)
	b := 0.
	E := make([]float64, m, m)
	eta := 0.
	passes := 0
	L, H := 0., 0.
	type KcacheEntry struct {
		i, j int
		v    float64
	}
	Kcache := make([]KcacheEntry, 0)

	K := func(i, j int) float64 {
		if i > j {
			i, j = j, i
		}
		for off, e := range Kcache {
			if e.i == i && e.j == j {
				copy(Kcache[1:off+1], Kcache[0:off])
				Kcache[0] = e
				return e.v
			}
		}
		if len(Kcache) > CacheSize/8 {
			Kcache = Kcache[0 : CacheSize/8]
		}
		e := KcacheEntry{i, j, KernelFunction(X.RawRowView(i), X.RawRowView(j))}
		Kcache = append([]KcacheEntry{e}, Kcache...)
		return e.v
	}
	calcE := func(i int) {
		//Calculate Ei = f(x(i)) - y(i) using E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
		sumAlphaYK := 0.
		for i1 := 0; i1 < m; i1++ {
			sumAlphaYK += alphas[i1] * Y[i1] * K(i1, i)
		}
		E[i] = b + sumAlphaYK - Y[i]
	}
	for passes < MaxPasses {
		numChangedAlphas := 0
		for i := 0; i < m; i++ {
			calcE(i)
			if (Y[i]*E[i] < -Tol && alphas[i] < C) || (Y[i]*E[i] > Tol && alphas[i] > 0) {
				// % In practice, there are many heuristics one can use to select
				// % the i and j. In this simplified code, we select them randomly.
				j := rand.Intn(m - 1)
				if j >= i {
					j++
				}
				calcE(j)

				alphaiold, alphajold := alphas[i], alphas[j]
				//% Compute L and H by (10) or (11).
				if Y[i] == Y[j] {
					L, H = math.Max(0, alphas[j]+alphas[i]-C), math.Min(C, alphas[j]+alphas[i])
				} else {
					L, H = math.Max(0, alphas[j]-alphas[i]), math.Min(C, C+alphas[j]-alphas[i])
				}
				if L == H {
					continue
				}
				Kij := K(i, j)
				Kii := K(i, i)
				Kjj := K(j, j)
				eta = 2*Kij - Kii - Kjj
				if eta >= 0 {
					continue
				}
				// % Compute and clip new value for alpha j using (12) and (15).
				alphas[j] = alphas[j] - (Y[j]*(E[i]-E[j]))/eta

				// % Clip
				alphas[j] = math.Max(L, math.Min(H, alphas[j]))
				// % Check if change in alpha is significant
				if alphas[j]-alphajold < Tol {
					alphas[j] = alphajold
					continue
				}
				// % Determine value for alpha i using (16).
				alphas[i] += Y[i] * Y[j] * (alphajold - alphas[j])
				// % Compute b1 and b2 using (17) and (18) respectively.
				b1 := b - E[i] - Y[i]*(alphas[i]-alphaiold)*Kij - Y[j]*(alphas[j]-alphajold)*Kij
				b2 := b - E[j] - Y[i]*(alphas[i]-alphaiold)*Kij - Y[j]*(alphas[j]-alphajold)*Kjj
				// % Compute b by (19).
				if 0 < alphas[i] && alphas[i] < C {
					b = b1
				} else if 0 < alphas[j] && alphas[j] < C {
					b = b2
				} else {
					b = (b1 + b2) / 2
				}
				numChangedAlphas++
			}
		}
		if numChangedAlphas == 0 {
			passes++
		} else {
			passes = 0
		}
	}
	idx := make([]int, 0)
	for i := 0; i < m; i++ {
		if alphas[i] > 1e-15 {
			idx = append(idx, i)
		}
	}
	model := &Model{
		X:              mat.NewDense(len(idx), n, nil),
		Y:              make([]float64, len(idx)),
		KernelFunction: KernelFunction,
		B:              b,
		Alphas:         make([]float64, len(idx)),
		W:              make([]float64, n, n),
	}
	for ii, i := range idx {
		model.X.SetRow(ii, X.RawRowView(i))
		model.Y[ii] = Y[i]
		model.Alphas[ii] = alphas[i]

	}
	for j := 0; j < n; j++ {
		for _, i := range idx {
			model.W[j] += alphas[i] * Y[i] * X.At(i, j)
		}
	}
	return model
}

// %svmPredict returns a vector of predictions using a trained SVM model
// %(svmTrain).
// %   pred = SVMPREDICT(model, X) returns a vector of predictions using a
// %   trained SVM model (svmTrain). X is a mxn matrix where there each
// %   example is a row. model is a svm model returned from svmTrain.
// %   predictions pred is a m x 1 column of predictions of {0, 1} values.
func svmPredict(model *Model, X *mat.Dense) (pred []float64) {
	m, _ := X.Dims()
	p := make([]float64, m, m)
	pred = make([]float64, m, m)
	for i := 0; i < m; i++ {
		prediction := 0.
		for j := range model.Alphas {
			prediction += model.Alphas[j] * model.Y[j] * model.KernelFunction(X.RawRowView(i), model.X.RawRowView(j))
		}
		p[i] = prediction + model.B
		if p[i] >= 0 {
			pred[i] = 1.
		}
	}
	return

}

// Kernel is the interface for kernels
type Kernel interface {
	Func(a, b []float64) float64
}

// LinearKernel is dot product
type LinearKernel struct{}

// Func for LinearKernel
func (LinearKernel) Func(a, b []float64) (sumprod float64) {
	for i := range a {
		sumprod += a[i] * b[i]
	}
	return
}

// PolynomialKernel ...
type PolynomialKernel struct{ gamma, coef0, degree float64 }

// Func for PolynomialKernel
func (kdata PolynomialKernel) Func(a, b []float64) (sumprod float64) {
	for i := range a {
		sumprod += a[i] * b[i]
	}
	return math.Pow(kdata.gamma*sumprod+kdata.coef0, kdata.degree)
}

// RBFKernel ...
type RBFKernel struct{ gamma float64 }

// Func for RBFKernel
func (kdata RBFKernel) Func(a, b []float64) float64 {
	L2 := 0.
	for i := range a {
		v := a[i] - b[i]
		L2 += v * v
	}
	return math.Exp(-kdata.gamma * L2)
}

// SigmoidKernel ...
type SigmoidKernel struct{ gamma, coef0 float64 }

// Func for SigmoidKernel
func (kdata SigmoidKernel) Func(a, b []float64) (sumprod float64) {
	for i := range a {
		sumprod += a[i] * b[i]
	}
	return math.Tanh(kdata.gamma*sumprod + kdata.coef0)
}

// SVC struct
type SVC struct {
	C           float64
	Kernel      interface{}
	Degree      float64
	Gamma       float64
	Coef0       float64
	Shrinking   bool
	Probability bool
	Tol         float64
	CacheSize   int
	ClassWeight []float64
	MaxIter     int
	Model       []*Model
}

// NewSVC ...
func NewSVC() *SVC {
	m := &SVC{
		C: 1., Kernel: "rbf", Degree: 3., Gamma: 0., Coef0: 0., Shrinking: true, Probability: false, Tol: 1e-3, CacheSize: 200,
	}
	return m
}

// Fit for SVC
func (m *SVC) Fit(X, Y *mat.Dense) base.Transformer {
	NSamples, NFeatures := X.Dims()
	_, Noutputs := Y.Dims()
	if m.Gamma <= 0. {
		m.Gamma = 1. / float64(NFeatures)
	}
	m.Model = make([]*Model, Noutputs)
	var K func(a, b []float64) float64
	switch v := m.Kernel.(type) {
	case func(a, b []float64) float64:
		K = v
	case string:
		switch v {
		case "linear":
			K = (LinearKernel{}).Func
		case "poly", "polynomial":
			K = (PolynomialKernel{gamma: m.Gamma, coef0: m.Coef0, degree: m.Degree}).Func
		case "rbf":
			K = (RBFKernel{gamma: m.Gamma}).Func
		case "sigmoid":
			K = (SigmoidKernel{gamma: m.Gamma, coef0: m.Coef0}).Func
		case "default":
			panic(fmt.Errorf("unknown kernel %s", v))
		}
	case Kernel:
		K = v.Func
	default:
		panic(fmt.Errorf("unknown kernel %#v", v))
	}
	base.Parallelize(-1, Noutputs, func(th, start, end int) {
		y := make([]float64, NSamples)
		for output := start; output < end; output++ {
			mat.Col(y, output, Y)
			m.Model[output] = svmTrain(X, y, m.C, K, m.Tol, m.MaxIter, m.CacheSize)

		}
	})
	return m
}

// Predict for SVC
func (m *SVC) Predict(X, Y *mat.Dense) base.Transformer {
	_, NOutputs := Y.Dims()
	if NOutputs == 0 {
		NSamples, _ := X.Dims()
		NOutputs = len(m.Model)
		*Y = *mat.NewDense(NSamples, NOutputs, nil)
	}
	base.Parallelize(-1, NOutputs, func(th, start, end int) {

		for output := start; output < end; output++ {

			pred := svmPredict(m.Model[output], X)
			Y.SetCol(output, pred)
		}
	})
	return m
}

// Transform for SVC for pipeline
func (m *SVC) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, _ := X.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, len(m.Model), nil)
	m.Predict(X, Yout)
	return
}
