package neighbors

import (
	"math"
	"runtime"
	"sort"
	"strings"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

// NearestNeighbors is the unsupervised alog implementing search of k nearest neighbors
// Algorithm is one of 'auto', 'ball_tree', 'kd_tree', 'brute' defaults to "auto"
//
// Metric = 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan' defaults to euclidean (= minkowski with P=2)
// P is power for 'minkowski'
// NJobs: number of concurrent jobs. NJobs<0 means runtime.NumCPU()  default to -1
type NearestNeighbors struct {
	Algorithm string
	Metric    string
	P         float64
	NJobs     int
	LeafSize  int
	// Runtime filled members
	Distance func(a, b mat.Vector) float64
	X, Y     *mat.Dense
	Tree     *KDTree
}

// NewNearestNeighbors returns an *NearestNeighbors
func NewNearestNeighbors() *NearestNeighbors {
	return &NearestNeighbors{Algorithm: "auto", Metric: "euclidean", P: 2, NJobs: -1}
}

// Fit for NearestNeighbors. Y is unused
func (m *NearestNeighbors) Fit(X, Y mat.Matrix) {
	r, c := X.Dims()
	switch m.Metric {
	case "manhattan", "cityblock":
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
	useKDTree := strings.Contains(strings.ToLower(m.Algorithm), "tree") || (m.Algorithm == "auto" && r*c > 1000)
	if useKDTree {
		if m.LeafSize <= 0 {
			m.LeafSize = 30
		}
		m.Tree = NewKDTree(X, m.LeafSize)
	}
}

// KNeighbors returns distances and indices of first NNeighbors
func (m *NearestNeighbors) KNeighbors(X mat.Matrix, NNeighbors int) (distances, indices *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	if m.Tree != nil {
		return m.Tree.Query(X, NNeighbors, 1e-15, m.P, math.Inf(1))
	}
	distances = mat.NewDense(NSamples, NNeighbors, nil)
	indices = mat.NewDense(NSamples, NNeighbors, nil)
	base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
		Xsample := mat.NewVecDense(NFeatures, nil)
		NFitSamples, _ := m.X.Dims()
		idx := make([]int, NFitSamples)
		sampleDistance := make([]float64, NFitSamples)

		for sample := start; sample < end; sample++ {

			mat.Row(Xsample.RawVector().Data, sample, X)
			base.Parallelize(m.NJobs, NFitSamples, func(th, start, end int) {
				for ifs := start; ifs < end; ifs++ {
					sampleDistance[ifs] = m.Distance(Xsample, m.X.RowView(ifs))
					idx[ifs] = ifs
				}
			})
			sort.Slice(idx, func(i, j int) bool { return sampleDistance[idx[i]] < sampleDistance[idx[j]] })
			for ik := 0; ik < NNeighbors; ik++ {
				indices.Set(sample, ik, float64(idx[ik]))
				distances.Set(sample, ik, sampleDistance[idx[ik]])
			}
		}
	})
	return
}

// KNeighborsGraph Computes the (weighted) graph of k-Neighbors for points in X
// mode : {‘connectivity’, ‘distance’}, optional
//     Type of returned matrix: ‘connectivity’ will return the connectivity matrix with ones and zeros, in ‘distance’ the edges are Euclidean distance between points.
// Returns:
// A : shape = [n_samples, n_samples_fit]
//     n_samples_fit is the number of samples in the fitted data A[i, j] is assigned the weight of edge that connects i to j.
func (m *NearestNeighbors) KNeighborsGraph(X *mat.Dense, NNeighbors int, mode string, includeSelf bool) (graph *mat.Dense) {
	NSamples, _ := X.Dims()
	NSamplesFit, _ := m.X.Dims()
	distances, indices := m.KNeighbors(X, NNeighbors)
	graph = mat.NewDense(NSamples, NSamplesFit, nil)
	var source *mat.Dense

	switch mode {
	case "connectivity":
		source = indices
	default:
		source = distances
	}
	base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
		for sample := start; sample < end; sample++ {
			for ik := 0; ik < NNeighbors; ik++ {
				index := int(source.At(sample, ik))
				if sample == index && !includeSelf {
					continue
				}
				graph.Set(sample, index, 1.)
			}
		}
	})
	return
}

// RadiusNeighbors Finds the neighbors within a given radius of a point or points.
// Return the indices and distances of each point from the dataset
// lying in a ball with size ``radius`` around the points of the query
// array. Points lying on the boundary are included in the results.
// The result points are *not* necessarily sorted by distance to their
// query point.
// Parameters
// ----------
// X : array-like, (n_samples, n_features), optional
// 	The query point or points.
// 	If not provided, neighbors of each indexed point are returned.
// 	In this case, the query point is not considered its own neighbor.
// radius : float
// 	Limiting distance of neighbors to return.
// 	(default is the value passed to the constructor).
func (m *NearestNeighbors) RadiusNeighbors(X *mat.Dense, radius float64) (distances [][]float64, indices [][]int) {
	NSamples, _ := X.Dims()
	distances = make([][]float64, NSamples)
	indices = make([][]int, NSamples)
	NFitSamples, _ := m.X.Dims()
	if m.Tree == nil {
		Mdistances, Mindices := m.KNeighbors(X, NFitSamples)
		base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
			for sample := start; sample < end; sample++ {
				for j := 0; j < NFitSamples; j++ {
					d := Mdistances.At(sample, j)
					if d > radius {
						break
					}
					distances[sample] = append(distances[sample], d)
					indices[sample] = append(indices[sample], int(Mindices.At(sample, j)))
				}
			}
		})
	} else {
		base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
			for sample := start; sample < end; sample++ {
				callback := func(ik int, dist float64, ind int) {
					if dist <= radius {
						distances[sample] = append(distances[sample], dist)
						indices[sample] = append(indices[sample], ind)
					}

				}
				m.Tree._query(X.RowView(sample), NFitSamples, 0, m.P, radius, callback)
			}
		})
	}
	return
}
