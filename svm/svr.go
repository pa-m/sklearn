package svm

import (
	"math"

	"golang.org/x/exp/rand"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

// SVR struct
type SVR struct {
	BaseLibSVM

	ClassWeight []float64
	nOutputs    int
}

// NewSVR ...
// Kernel: "linear","poly","rbf","sigmoid" default is "rbf"
// if Gamma<=0 il will be changed to 1/NFeatures
// Cachesize is in MB. defaults to 200
func NewSVR() *SVR {
	m := &SVR{
		BaseLibSVM: BaseLibSVM{C: 1., Epsilon: 0.1, Kernel: "rbf", Degree: 3., Gamma: 0., Coef0: 0., Shrinking: true, Tol: 1e-3, CacheSize: 200},
	}
	return m
}

// IsClassifier returns false for SVR
func (*SVR) IsClassifier() bool { return false }

// PredicterClone for SVR
func (m *SVR) PredicterClone() base.Predicter {
	if m == nil {
		return nil
	}
	clone := *m
	if sourceCloner, ok := clone.RandomState.(base.SourceCloner); ok && sourceCloner != base.SourceCloner(nil) {
		clone.RandomState = sourceCloner.SourceClone()
	}
	return &clone
}

func svrTrain(X *mat.Dense, Y []float64, C, Epsilon float64, KernelFunction func(X1, X2 []float64) float64, Tol float64, MaxPasses int, CacheSize uint, RandomState base.Source) *Model {
	m, n := X.Dims()
	alphas := make([]float64, m)
	b := 0.
	E := make([]float64, m)
	eta := 0.
	passes := 0
	L, H := 0., 0.
	K := cachedKernel(X, CacheSize, KernelFunction)
	f := func(i int) float64 {
		y := b
		for i1 := 0; i1 < m; i1++ {
			y += alphas[i1] * K(i1, i)
		}
		return y
	}
	sgn := func(x float64) float64 {
		if x >= 0 {
			return 1
		}
		return -1
	}
	abs := math.Abs
	max := math.Max
	min := math.Min
	step := func(x float64) float64 {
		if x < 0 {
			return 0
		}
		return 1
	}
	randIntn := rand.Intn
	if RandomState != nil {
		type Intner interface{ Intn(int) int }
		if intner, ok := RandomState.(Intner); ok {
			randIntn = intner.Intn
		} else {
			randIntn = rand.New(RandomState).Intn
		}
	}

	for passes < MaxPasses {
		numChangedAlphas := 0
		// Step 1 Find a Lagrange multiplier α 1 {that violates the Karush–Kuhn–Tucker (KKT) conditions for the optimization problem.
		var KKTviolated bool
		for sample := 0; sample < m; sample++ {
			i := sample
			E[i] = f(i) - Y[i]
			if (E[i] < -Epsilon && alphas[i] < C) || (E[i] > Epsilon && alphas[i] > -C) {
				KKTviolated = true
				// Step 2 Pick a second multiplier α 2  and optimize the pair ( α 1 , α 2 )
				// % In practice, there are many heuristics one can use to select
				// % the i and j. In this simplified code, we select them randomly.
				j := randIntn(m - 1)
				if j >= i {
					j++
				}
				for j2 := 0; j2 < m; j2++ {
					j3 := (j + j2) % m
					if j3 == i {
						continue
					}
					E[j3] = f(j3) - Y[j3]
					if (E[j3] < -Epsilon) || (E[j] > Epsilon) {
						j = j3
						break
					}
				}
				//E[j] = f(j) - Y[j]
				Kii := K(i, i)
				Kij := K(i, j)
				Kjj := K(j, j)
				eta = 2*Kij - Kii - Kjj
				alphaiold, alphajold := alphas[i], alphas[j]
				s := alphaiold + alphajold
				delta := -2 * Epsilon / eta
				alphas[j] = alphajold - (E[i]-E[j])/eta
				alphas[i] = s - alphas[j]

				// if abs(alphas[j]) > abs(alphas[i]) {
				// 	E[i], E[j] = E[j], E[i]
				// 	Kij, Kjj = Kjj, Kii
				// 	alphaiold, alphajold = alphajold, alphaiold
				// 	alphas[i], alphas[j] = alphas[j], alphas[i]
				// }

				if alphas[i]*alphas[j] < 0 {
					aai, aaj := abs(alphas[i]), abs(alphas[j])
					if aai > delta || aaj > delta {
						alphas[j] -= sgn(alphas[j]) * delta
					} else {
						alphas[j] = step(aaj-aai) * s
					}
				}
				//% Compute L and H by (10) or (11).
				L, H = max(s-C, -C), min(C, C+s)
				alphas[j] = min(H, max(L, alphas[j]))
				alphas[i] = s - alphas[j]
				Einew := E[i] + (alphas[i]-alphaiold)*Kii + (alphas[j]-alphajold)*Kij
				Ejnew := E[j] + (alphas[i]-alphaiold)*Kij + (alphas[j]-alphajold)*Kjj
				db := -(Einew + Ejnew) / 2
				Einew, Ejnew = Einew+db, Ejnew+db
				Edecrease := abs(E[i])+abs(E[j]) > abs(Einew)+abs(Ejnew)

				// % Check if change in alpha is significant
				if abs(alphas[j]-alphajold) < Tol || !Edecrease {
					alphas[i], alphas[j] = alphaiold, alphajold
					continue
				}
				// log.Printf("%d,%d\tEdecrease = %.3f\n", i, j, Edecrease)
				b += db

				numChangedAlphas++
			}
			// Step 3: Repeat steps 1 and 2 until convergence.
		}
		if !KKTviolated {
			break
		}
		if numChangedAlphas == 0 {
			passes++
		} else {
			passes = 0
		}
	}
	idx := make([]int, 0)
	for i := 0; i < m; i++ {
		if abs(alphas[i]) > Tol {
			idx = append(idx, i)
		}
	}

	model := &Model{
		X:              mat.NewDense(len(idx), n, nil),
		KernelFunction: KernelFunction,
		B:              b,
		Alphas:         make([]float64, len(idx)),
		Support:        idx,
	}
	for ii, i := range idx {
		model.X.SetRow(ii, X.RawRowView(i))
		model.Alphas[ii] = alphas[i]
	}
	return model
}

// Fit for SVR
func (m *SVR) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	_, m.nOutputs = Ymatrix.Dims()
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	m.BaseLibSVM.fit(X, Y, svrTrain)
	return m
}

// GetNOutputs ...
func (m *SVR) GetNOutputs() int { return m.nOutputs }

func svrPredict(model *Model, X, Y *mat.Dense, output int) {
	NSamples, _ := X.Dims()

	Ymat := Y.RawMatrix()
	for i, yoff := 0, output; i < NSamples; i, yoff = i+1, yoff+Ymat.Stride {
		y := model.B
		for j := range model.Alphas {
			y += model.Alphas[j] * model.KernelFunction(X.RawRowView(i), model.X.RawRowView(j))
		}
		Ymat.Data[yoff] = y
	}
}

// Predict for SVR
func (m *SVR) Predict(Xmatrix mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}
	base.Parallelize(-1, m.GetNOutputs(), func(th, start, end int) {

		for output := start; output < end; output++ {

			svrPredict(m.Model[output], X, Y, output)

		}
	})
	return base.FromDense(Ymutable, Y)
}

// Score for SVR returns R2Score
func (m *SVR) Score(X, Y mat.Matrix) float64 {
	Ypred := m.Predict(X, nil)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
