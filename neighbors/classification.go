package neighbors

import (
	"fmt"
	"runtime"
	"sort"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"

	"gonum.org/v1/gonum/mat"
)

// KNeighborsClassifier is a Regression based on k-nearest neighbors.
// The target is predicted by local interpolation of the targets
// associated of the nearest neighbors in the training set.
type KNeighborsClassifier struct {
	base.Predicter
	NearestNeighbors
	K        int
	Weight   string
	Scale    bool
	Distance Distance
	// Runtime members
	Xscaled, Y *mat.Dense
	Classes    [][]float64
	nOutputs   int
}

// NewKNeighborsClassifier returns an initialized *KNeighborsClassifier
func NewKNeighborsClassifier(K int, Weights string) *KNeighborsClassifier {
	return &KNeighborsClassifier{NearestNeighbors: *NewNearestNeighbors(), K: K, Weight: Weights}
}

// Fit ...
func (m *KNeighborsClassifier) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	m.Xscaled = mat.DenseCopyOf(X)
	m.Y = Y
	m.nOutputs = Y.RawMatrix().Cols
	if m.Distance == nil {
		m.Distance = EuclideanDistance
	}
	if m.K <= 0 {
		panic(fmt.Errorf("K<=0"))
	}
	m.NearestNeighbors.Fit(X, Y)
	m.Classes, _ = getClasses(Y)
	return m
}

// GetNOutputs returns output columns number for Y to pass to predict
func (m *KNeighborsClassifier) GetNOutputs() int {
	return m.nOutputs
}

// Predict  for KNeighborsClassifier
func (m *KNeighborsClassifier) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}

	m._predict(base.ToDense(X), Y, false)
	return base.FromDense(Ymutable, Y)
}

// PredictProba for KNeighborsClassifier
func (m *KNeighborsClassifier) PredictProba(X, Y *mat.Dense) *KNeighborsClassifier {
	return m._predict(X, Y, true)
}

func (m *KNeighborsClassifier) _predict(X, Y *mat.Dense, wantProba bool) *KNeighborsClassifier {
	_, outputs := m.Y.Dims()
	if wantProba {
		if outputs > 1 {
			panic("PredictProba is undefined for multioutput classification")
		}
		if Y == nil {
			panic("Y is unallocated")
		}
		_, Ycols := Y.Dims()
		if Ycols != len(m.Classes[0]) {
			panic(fmt.Errorf("PredictProba theres %d classes but Y has %d columns", len(m.Classes[0]), Ycols))
		}
	}
	NX, _ := X.Dims()

	NPROCS := runtime.GOMAXPROCS(0)
	isWeightDistance := m.Weight == "distance"
	distances, indices := m.KNeighbors(X, m.K)

	base.Parallelize(NPROCS, NX, func(th, start, end int) {
		epsilon := 1e-15
		weights := make([]float64, m.K)
		sumweights := 0.
		ys := make([]float64, m.K)
		if !isWeightDistance {
			for ik := range weights {
				weights[ik] = 1.
				sumweights += 1.
			}
		}
		for sample := start; sample < end; sample++ {
			// set Y(sample,output) to weighted average of K nearest

			for o := 0; o < outputs; o++ {
				classw := make(map[float64]float64)
				if isWeightDistance {
					sumweights = 0.
				}
				for ik := range ys {
					cl := m.Y.At(int(indices.At(sample, ik)), o)
					if isWeightDistance {
						dist := distances.At(sample, ik)
						weights[ik] = 1. / (epsilon + dist)
						sumweights += weights[ik]

					}
					if clw, present := classw[cl]; present {
						classw[cl] = clw + weights[ik]
					} else {
						classw[cl] = weights[ik]
					}
				}
				wmax, clwmax := 0., 0.
				for cl, w := range classw {
					if w > wmax {
						wmax = w
						clwmax = cl
					}
				}
				if wantProba {
					for icl, cl := range m.Classes[0] {
						if clw, found := classw[cl]; found {
							Y.Set(sample, icl, clw/sumweights)
						}
					}
				} else {
					Y.Set(sample, o, clwmax)

				}
			}
		}
	})
	return m
}

// Score for KNeighborsClassifier
func (m *KNeighborsClassifier) Score(X, Y mat.Matrix) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.AccuracyScore(Y, Ypred, true, nil)
}

func getClasses(Y mat.Matrix) (classes [][]float64, counts [][]int) {
	NSamples, Noutputs := Y.Dims()
	for o := 0; o < Noutputs; o++ {
		clmap := make(map[float64]bool)
		clcnt := make(map[float64]int)
		for s := 0; s < NSamples; s++ {
			cl := Y.At(s, o)
			clmap[cl] = true
			if cnt, present := clcnt[cl]; present {
				clcnt[cl] = cnt + 1
			} else {
				clcnt[cl] = 1
			}
		}
		clvalues := make([]float64, 0)
		clcounts := make([]int, 0)
		for cl := range clmap {
			clvalues = append(clvalues, cl)
			clcounts = append(clcounts, clcnt[cl])
		}
		sort.Float64s(clvalues)
		classes = append(classes, clvalues)
		counts = append(counts, clcounts)
	}
	return
}
