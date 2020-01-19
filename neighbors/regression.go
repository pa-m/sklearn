package neighbors

import (
	"fmt"
	"runtime"

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
	K int
	// Weights may be "uniform", "distance" or func(dstWeights, srcDists []float64)
	Weights  interface{}
	Scale    bool
	Distance Distance
	// Runtime members
	Xscaled, Y *mat.Dense
}

// NewKNeighborsRegressor returns an initialized *KNeighborsRegressor
// Weights may be "uniform", "distance" or func(dist []float64) []float64
func NewKNeighborsRegressor(K int, Weights interface{}) base.Predicter {
	return &KNeighborsRegressor{NearestNeighbors: *NewNearestNeighbors(), K: K, Weights: Weights}
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
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}

	NX, _ := X.Dims()
	_, outputs := m.Y.Dims()

	NPROCS := runtime.GOMAXPROCS(0)
	var isWeightDistance bool
	type wfntype = func(dstWeights, srcDists []float64)
	var wfn wfntype
	if weightsstr, ok := m.Weights.(string); ok && weightsstr == "distance" {
		isWeightDistance = true
	} else if tmpfn, ok := m.Weights.(wfntype); ok {
		wfn = tmpfn
	}

	distances, indices := m.KNeighbors(X, m.K)

	base.Parallelize(NPROCS, NX, func(th, start, end int) {
		weights := make([]float64, m.K)
		dists := make([]float64, m.K)
		ys := make([]float64, m.K)
		epsilon := 1e-15
		for ik := range weights {
			weights[ik] = 1.
		}
		for sample := start; sample < end; sample++ {
			// set Y(sample,output) to weighted average of K nearest
			mat.Row(dists, sample, distances)
			for o := 0; o < outputs; o++ {
				for ik := range ys {
					ys[ik] = m.Y.At(int(indices.At(sample, ik)), o)
					if wfn != nil {
						wfn(weights, dists)
					} else if isWeightDistance {
						weights[ik] = 1. / (epsilon + dists[ik])
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
