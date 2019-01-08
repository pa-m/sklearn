package cluster

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

// KMeans grouping algo
type KMeans struct {
	// Required members
	NClusters int
	// Optional members
	NJobs    int
	Distance func(X, Y mat.Vector) float64
	// Runtime filled members
	Centroids *mat.Dense
}

// Clone for KMeans
func (m *KMeans) Clone() base.Transformer {
	clone := *m
	return &clone
}

// Nearest returns index of the nearest centroid
func (m *KMeans) _nearest(scaledRow mat.Vector) int {
	best := -1
	var bestDist float64
	for ic := 0; ic < m.NClusters; ic++ {
		d := m.Distance(scaledRow, m.Centroids.RowView(ic))
		if best < 0 || d < bestDist {
			best = ic
			bestDist = d
		}
	}
	return best
}

// Fit compute centroids
// Y is useless here but we want all classifiers have the same interface. pass nil
func (m *KMeans) Fit(X, Y *mat.Dense) base.Transformer {
	NSamples, NFeatures := X.Dims()
	if NSamples < m.NClusters {
		panic(fmt.Errorf("NSamples<m.NClusters %d<%d", NSamples, m.NClusters))
	}
	if m.Distance == nil {
		m.Distance = EuclideanDistance

	}

	m.Centroids = mat.NewDense(m.NClusters, NFeatures, nil)
	row := make([]float64, NFeatures, NFeatures)
	for ic := 0; ic < m.NClusters; ic++ {
		mat.Row(row, ic, X)
		m.Centroids.SetRow(ic, row)
	}
	NearestCentroid := make([]int, NSamples)
	CentroidCount := make([]int, m.NClusters)
	epoch := 0
	changed := true
	unchangeCount := 0

	if m.NJobs <= 0 {
		m.NJobs = runtime.NumCPU()
	}

	for unchangeCount < 2 {
		epoch++
		changed = false
		// find nearest centroids
		m.predictInternal(X, NearestCentroid, CentroidCount, &changed)
		// recompute centroids
		m.Centroids.Sub(m.Centroids, m.Centroids)
		var mu sync.Mutex
		base.Parallelize(m.NJobs, NSamples, func(th, start, end int) {
			row := make([]float64, NFeatures, NFeatures)
			for sample := start; sample < end; sample++ {
				ic := NearestCentroid[sample]
				mu.Lock()
				c := m.Centroids.RowView(ic)
				mat.Row(row, sample, X)
				c.(*mat.VecDense).AddScaledVec(c, 1./float64(CentroidCount[ic]), mat.NewVecDense(NFeatures, row))
				mu.Unlock()
			}
		})
		if changed {
			unchangeCount = 0
		} else {
			unchangeCount++
		}
	}
	return m
}

// Transform for pipeline
func (m *KMeans) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, _ := X.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, 1, nil)
	m.Predict(X, Yout)
	return
}

func (m *KMeans) predictInternal(Xscaled mat.Matrix, y, CentroidCount []int, changed *bool) {
	NSamples, NFeatures := Xscaled.Dims()
	if y == nil {
		y = make([]int, NSamples, NSamples)
	}
	if CentroidCount != nil {
		for ic := range CentroidCount {
			CentroidCount[ic] = 0
		}
	}
	var m1, m2 sync.Mutex
	base.Parallelize(runtime.NumCPU(), NSamples, func(th, start, end int) {
		var row mat.Vector
		var Xrv mat.RowViewer
		var isrv bool
		if Xrv, isrv = Xscaled.(mat.RowViewer); !isrv {
			row = mat.NewVecDense(NFeatures, nil)
		}
		for sample := start; sample < end; sample++ {
			if isrv {
				row = Xrv.RowView(sample)
			} else {
				mat.Row(row.(mat.RawVectorer).RawVector().Data, sample, Xscaled)
			}
			nearest := m._nearest(row)
			if nearest != y[sample] {
				y[sample] = nearest
				if changed != nil {
					m1.Lock()
					*changed = true
					m1.Unlock()
				}
			}
			if CentroidCount != nil {
				m2.Lock()
				CentroidCount[nearest]++
				m2.Unlock()
			}
		}
	})
}

// Predict fills y with indices of centroids
func (m *KMeans) Predict(X, Y *mat.Dense) {
	NSamples, _ := X.Dims()
	y := make([]int, NSamples, NSamples)
	m.predictInternal(X, y, nil, nil)
	for i, y1 := range y {
		Y.Set(i, 0, float64(y1))
	}
}
