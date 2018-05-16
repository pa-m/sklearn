package neighbors

import (
	"fmt"
	"runtime"
	"sort"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// KNeighborsRegressor is a Regression based on k-nearest neighbors.
// The target is predicted by local interpolation of the targets
// associated of the nearest neighbors in the training set.
type KNeighborsRegressor struct {
	NearestNeighbors
	K        int
	Weight   string
	Scale    bool
	Distance Distance
	// Runtime members
	Xscaled, Y *mat.Dense
}

// NewKNeighborsRegressor returns an initialized *KNeighborsRegressor
func NewKNeighborsRegressor(K int, Weights string) base.Regressor {
	return &KNeighborsRegressor{NearestNeighbors: *NewNearestNeighbors(), K: K, Weight: Weights}
}

// Fit ...
func (m *KNeighborsRegressor) Fit(X, Y *mat.Dense) base.Transformer {
	m.Xscaled = mat.DenseCopyOf(X)
	m.Y = mat.DenseCopyOf(Y)
	if m.Distance == nil {
		m.Distance = EuclideanDistance
	}
	if m.K <= 0 {
		panic(fmt.Errorf("K<=0"))
	}
	m.NearestNeighbors.Fit(X)
	return m
}

// Predict ...
func (m *KNeighborsRegressor) Predict(X, Y *mat.Dense) base.Regressor {
	NFitSamples, _ := m.Xscaled.Dims()
	NX, _ := X.Dims()
	_, outputs := m.Y.Dims()

	NCPU := runtime.NumCPU()
	isWeightDistance := m.Weight == "distance"
	distances, indices := m.KNeighbors(X, m.K)

	base.Parallelize(NCPU, NX, func(th, start, end int) {
		d2 := make([]float64, NFitSamples)
		idx := make([]int, NFitSamples)
		weights := make([]float64, m.K)
		ys := make([]float64, m.K)
		epsilon := 1e-15
		for ik := range weights {
			weights[ik] = 1.
		}
		for sample := start; sample < end; sample++ {
			// sort idx to get first K nearest
			sort.Slice(idx, func(i, j int) bool { return d2[idx[i]] < d2[idx[j]] })
			// set Y(sample,output) to weighted average of K nearest
			for o := 0; o < outputs; o++ {
				for ik := range ys {
					ys[ik] = m.Y.At(int(indices.At(sample, ik)), o)
					if isWeightDistance {
						weights[ik] = 1. / (epsilon + distances.At(sample, ik))
					}
				}
				Y.Set(sample, o, stat.Mean(ys, weights))
			}

		}
	})
	return m
}

// Transform for KNeighborsRegressor
func (m *KNeighborsRegressor) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, NOutputs := Y.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Yout)
	return
}

// Score for KNeighborsRegressor
func (m *KNeighborsRegressor) Score(X, Y *mat.Dense) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
