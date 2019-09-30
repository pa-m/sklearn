package neighbors

import (
	"runtime"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// NearestCentroid is a Regression based on k-nearest neighbors.
// The target is predicted by local interpolation of the targets
// associated of the nearest neighbors in the training set.
type NearestCentroid struct {
	base.Predicter
	Metric          string
	ShrinkThreshold float64
	// runtime filled members
	NearestNeighbors
	Classes    [][]float64
	ClassCount [][]int
	Centroids  *mat.Dense
}

// NewNearestCentroid ...
// if Metric is "manhattan", centroids are computed using median else mean
func NewNearestCentroid(metric string, shrinkThreshold float64) *NearestCentroid {
	return &NearestCentroid{Metric: metric, ShrinkThreshold: shrinkThreshold}
}

// Fit ...
func (m *NearestCentroid) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()
	if NOutputs != 1 {
		panic("NearestCentroid can't handle output Dim != 1")
	}
	m.Classes, m.ClassCount = getClasses(Y)
	NClasses := len(m.Classes[0])
	Centroids := mat.NewDense(NClasses, NFeatures, nil)
	useMedian := m.Metric == "manhattan"
	base.Parallelize(runtime.NumCPU(), NClasses*NFeatures, func(th, start, end int) {
		var centroidXfeat, vclass float64
		var icl, feature, rows, sample int
		col := make([]float64, NSamples)
		for index := start; index < end; index++ {
			icl = index / NFeatures
			feature = index % NFeatures
			rows = 0
			vclass = m.Classes[0][icl]
			for sample = 0; sample < NSamples; sample++ {
				if Y.At(sample, 0) == vclass {
					col[rows] = X.At(sample, feature)
					rows++
				}
			}
			if useMedian {
				centroidXfeat = stat.Quantile(.5, stat.Empirical, col[0:rows], nil)
			} else {
				centroidXfeat = stat.Mean(col[0:rows], nil)
			}
			Centroids.Set(icl, feature, centroidXfeat)
		}
	})
	m.NearestNeighbors.Fit(Centroids, mat.Matrix(nil))
	return m
}

// Predict  for NearestCentroid
func (m *NearestCentroid) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}

	m._predict(base.ToDense(X), Y, false)
	return base.FromDense(Ymutable, Y)
}

// PredictProba for NearestCentroid
func (m *NearestCentroid) PredictProba(X mat.Matrix, Y *mat.Dense) *NearestCentroid {
	return m._predict(X, Y, true)
}

func (m *NearestCentroid) _predict(X mat.Matrix, Y *mat.Dense, wantProba bool) *NearestCentroid {
	if wantProba {
		panic("PredictProba is undefined for NearestCentroid")
	}
	NSamples, _ := X.Dims()
	_, indices := m.KNeighbors(X, 1)
	for sample := 0; sample < NSamples; sample++ {
		icentroid := int(indices.At(sample, 0))

		Y.Set(sample, 0, m.Classes[0][icentroid])
	}
	return m
}

// Score for NearestCentroid
func (m *NearestCentroid) Score(X, Y mat.Matrix) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.AccuracyScore(Y, Ypred, true, nil)
}
