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
func NewKNeighborsRegressor(K int, Weights string) base.Predicter {
	return &KNeighborsRegressor{NearestNeighbors: *NewNearestNeighbors(), K: K, Weight: Weights}
}

// PredicterClone return a (possibly unfitted) copy of predicter
func (m *KNeighborsRegressor) PredicterClone() base.Predicter {
	clone := *m
	return &clone
}

// IsClassifier returns false for KNeighborsRegressor
func (*KNeighborsRegressor) IsClassifier() bool { return false }

// Fit ...
func (m *KNeighborsRegressor) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	m.Xscaled = mat.DenseCopyOf(X)
	m.Y = mat.DenseCopyOf(Y)
	if m.Distance == nil {
		m.Distance = EuclideanDistance
	}
	if m.K <= 0 {
		panic(fmt.Errorf("K<=0"))
	}
	m.NearestNeighbors.Fit(X, Y)
	return m
}

// GetNOutputs return Y width
func (m *KNeighborsRegressor) GetNOutputs() int { return m.Y.RawMatrix().Cols }

// Predict ...
func (m *KNeighborsRegressor) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsZero() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}

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
	return base.FromDense(Ymutable, Y)
}

// Score for KNeighborsRegressor
func (m *KNeighborsRegressor) Score(X, Y mat.Matrix) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
