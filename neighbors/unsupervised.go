package neighbors

import (
	"runtime"
	"sort"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

// NearestNeighbors is the unsupervised alog implementing search of k nearest neighbors
// Algorithm is one of 'auto', 'ball_tree', 'kd_tree', 'brute' defaults to "auto"
// Metric = 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan' defaults to euclidean (= minkowski with P=2)
// P is power for 'minkowski'
// NJobs: number of concurrent jobs. NJobs<0 means runtime.NumCPU()  default to -1
type NearestNeighbors struct {
	Algorithm string
	Metric    string
	P         float64
	NJobs     int
	// Runtime filled members
	Distance func(a, b mat.Vector) float64
	X, Y     *mat.Dense
}

// NewNearestNeighbors returns an *NearestNeighbors
func NewNearestNeighbors() *NearestNeighbors {
	return &NearestNeighbors{Algorithm: "auto", Metric: "euclidean", P: 2, NJobs: -1}
}

// Fit for NearestNeighbors
func (m *NearestNeighbors) Fit(X mat.Matrix) {
	switch m.Metric {
	case "manhattan":
		m.P = 1
		m.Distance = MinkowskiDistance(m.P)
	case "euclidean":
		m.P = 2
		m.Distance = MinkowskiDistance(m.P)
	default:
		m.Distance = MinkowskiDistance(m.P)
	}
	if m.NJobs < 0 {
		m.NJobs = runtime.NumCPU()
	}
	m.X = mat.DenseCopyOf(X)
}

// KNeighbors returns distances and indices of first NNeighbors
func (m *NearestNeighbors) KNeighbors(X mat.Matrix, NNeighbors int) (distances, indices *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	distances = mat.NewDense(NSamples, NNeighbors, nil)
	indices = mat.NewDense(NSamples, NNeighbors, nil)
	base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
		Xsample := mat.NewVecDense(NFeatures, nil)
		for sample := start; sample < end; sample++ {

			mat.Row(Xsample.RawVector().Data, sample, X)
			NFitSamples, _ := m.X.Dims()
			sampleDistance := make([]float64, NFitSamples, NFitSamples)
			idx := make([]int, NFitSamples, NFitSamples)
			base.Parallelize(m.NJobs, NFitSamples, func(th, start, end int) {
				for ifs := start; ifs < end; ifs++ {
					sampleDistance[ifs] = m.Distance(Xsample, m.X.RowView(ifs))
					idx[ifs] = ifs
				}
			})
			sort.Slice(idx, func(i, j int) bool { return sampleDistance[idx[i]] < sampleDistance[idx[j]] })
			for i := 0; i < NNeighbors; i++ {
				distances.Set(sample, i, sampleDistance[idx[i]])
				indices.Set(sample, i, float64(idx[i]))
			}
		}
	})
	return
}
