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
	base.Classifier
	NearestNeighbors
	K        int
	Weight   string
	Scale    bool
	Distance Distance
	// Runtime members
	Xscaled, Y *mat.Dense
	Classes    [][]float64
}

// NewKNeighborsClassifier returns an initialized *KNeighborsClassifier
func NewKNeighborsClassifier(K int, Weights string) *KNeighborsClassifier {
	return &KNeighborsClassifier{NearestNeighbors: *NewNearestNeighbors(), K: K, Weight: Weights}
}

// Fit ...
func (m *KNeighborsClassifier) Fit(X, Y *mat.Dense) base.Transformer {
	m.Xscaled = mat.DenseCopyOf(X)
	m.Y = mat.DenseCopyOf(Y)
	if m.Distance == nil {
		m.Distance = EuclideanDistance
	}
	if m.K <= 0 {
		panic(fmt.Errorf("K<=0"))
	}
	m.NearestNeighbors.Fit(X)
	NSamples, Noutputs := Y.Dims()
	for o := 0; o < Noutputs; o++ {
		clmap := make(map[float64]bool)
		for s := 0; s < NSamples; s++ {
			clmap[Y.At(s, o)] = true
		}
		clvalues := make([]float64, 0)
		for cl := range clmap {
			clvalues = append(clvalues, cl)
		}
		sort.Float64s(clvalues)
		m.Classes = append(m.Classes, clvalues)
	}
	return m
}

// Predict  for KNeighborsClassifier
func (m *KNeighborsClassifier) Predict(X, Y *mat.Dense) base.Transformer {
	return m._predict(X, Y, false)
}

// PredictProba for KNeighborsClassifier
func (m *KNeighborsClassifier) PredictProba(X, Y *mat.Dense) base.Transformer {
	return m._predict(X, Y, true)
}

func (m *KNeighborsClassifier) _predict(X, Y *mat.Dense, wantProba bool) base.Transformer {
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

	NCPU := runtime.NumCPU()
	isWeightDistance := m.Weight == "distance"
	distances, indices := m.KNeighbors(X, m.K)

	base.Parallelize(NCPU, NX, func(th, start, end int) {
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

// Transform for KNeighborsClassifier
func (m *KNeighborsClassifier) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, NOutputs := Y.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Yout)
	return
}

// Score for KNeighborsClassifier
func (m *KNeighborsClassifier) Score(X, Y *mat.Dense) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
