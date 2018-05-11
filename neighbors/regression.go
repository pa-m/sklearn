package neighbors

import (
	"fmt"
	"runtime"
	"sort"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"

	"github.com/gonum/stat"
	"gonum.org/v1/gonum/mat"
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
	return m
}

// Predict ...
func (m *KNeighborsRegressor) Predict(X, Y *mat.Dense) base.Regressor {
	NFitSamples, _ := m.Xscaled.Dims()
	NX, _ := X.Dims()
	_, outputs := m.Y.Dims()
	var Xscaled mat.Matrix
	Xscaled = X

	NCPU := runtime.NumCPU()
	isWeightDistance := m.Weight == "distance"
	base.Parallelize(NCPU, NX, func(th, start, end int) {
		var sumd2 float64
		d2 := make([]float64, NFitSamples)
		idx := make([]int, NFitSamples)
		weights := make([]float64, m.K)
		ys := make([]float64, m.K)
		for ik := range weights {
			weights[ik] = 1.
		}
		for sample := start; sample < end; sample++ {
			// fill d2 with square distances from sample to fit-samples
			base.Parallelize(NCPU, NFitSamples, func(th, start, end int) {
				for ifs := start; ifs < end; ifs++ {
					idx[ifs] = ifs
					d2[ifs] = m.Distance(Xscaled.(mat.RowViewer).RowView(sample), m.Xscaled.RowView(ifs))
					d2[ifs] *= d2[ifs]
				}
			})
			// sort idx to get first K nearest
			sort.Slice(idx, func(i, j int) bool { return d2[idx[i]] < d2[idx[j]] })
			// set Y(sample,output) to weighted average of K nearest
			sumd2 = 0.
			for ik := range ys {
				sumd2 += d2[idx[ik]]
			}
			for o := 0; o < outputs; o++ {
				for ik := range ys {
					ys[ik] = m.Y.At(idx[ik], o)
					if isWeightDistance {
						weights[ik] = 1. - d2[idx[ik]]/sumd2
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
