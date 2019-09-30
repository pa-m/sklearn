package cluster

import (
	"runtime"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/neighbors"
	"gonum.org/v1/gonum/mat"
)

// DBSCANConfig is the configuration structure for NewDBSCAN
type DBSCANConfig struct {
	Eps          float64
	MinSamples   float64
	Metric       string
	MetricsParam interface{}
	Algorithm    string
	LeafSize     int
	P            float64
	NJobs        int
}

// DBSCAN classifier struct
type DBSCAN struct {
	DBSCANConfig
	SampleWeight []float64
	// members filled by Fit
	NeighborsModel    *neighbors.NearestNeighbors
	Labels            []int
	CoreSampleIndices []int
}

// NewDBSCAN creates an *DBSCAN
// if config is nil, defaults are used
// defaults are Eps:.5 MinSamples:5 Metric:"euclidean" algorithm="auto" LeafSize:30 P:2 NJobs:runtime.NumCPU()
func NewDBSCAN(config *DBSCANConfig) *DBSCAN {
	if config == nil {
		config = &DBSCANConfig{}
	}
	if config.Eps <= 0 {
		config.Eps = 0.5
	}
	if config.MinSamples <= 0 {
		config.MinSamples = 5
	}
	if config.Metric == "" {
		config.Metric = "euclidean"
	}
	if config.Algorithm == "" {
		config.Algorithm = "auto"
	}
	if config.LeafSize <= 0 {
		config.LeafSize = 30
	}
	if config.P <= 0 {
		config.P = 2
	}
	if config.NJobs < 0 {
		config.NJobs = runtime.NumCPU()
	}
	return &DBSCAN{DBSCANConfig: *config}
}

// PredicterClone for DBSCAN
func (m *DBSCAN) PredicterClone() base.Predicter {
	clone := *m
	return &clone
}

// IsClassifier returns true for DBSCAN
func (m *DBSCAN) IsClassifier() bool { return true }

// Fit for DBSCAN
// X : mat.Dense of shape (n_samples, n_features)
// A feature array`.
// m.SampleWeight is used if not nil
// it is the Weight of each sample, such that a sample with a weight of at least
// ``min_samples`` is by itself a core sample; a sample with negative
// weight may inhibit its eps-neighbor from being core.
// Note that weights are absolute, and default to 1.
// Y : Ignored, may be nil
func (m *DBSCAN) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	m.NeighborsModel = neighbors.NewNearestNeighbors()
	m.NeighborsModel.Algorithm = m.Algorithm
	m.NeighborsModel.Distance = MinkowskiDistance(m.P)
	m.NeighborsModel.Metric = m.Metric
	m.NeighborsModel.NJobs = m.NJobs
	m.NeighborsModel.LeafSize = m.LeafSize
	m.NeighborsModel.Fit(X, nil)
	_, neighborhoods := m.NeighborsModel.RadiusNeighbors(X, m.Eps)

	NSamples, _ := X.Dims()
	NNeighbors := make([]float64, NSamples)

	if m.SampleWeight == nil {
		for i, neighbors := range neighborhoods {
			NNeighbors[i] = float64(len(neighbors))
		}
	} else {
		for i, neighbors := range neighborhoods {
			for _, neighbor := range neighbors {
				NNeighbors[i] += m.SampleWeight[neighbor]
			}
		}

	}
	// # Initially, all samples are noise.
	m.Labels = make([]int, NSamples)
	for i := range m.Labels {
		m.Labels[i] = -1
	}
	// # A list of all core samples found.
	isCore := make([]bool, NSamples)
	for sample := range NNeighbors {
		if NNeighbors[sample] > m.MinSamples {
			isCore[sample] = true
			m.CoreSampleIndices = append(m.CoreSampleIndices, sample)
		}
	}
	dbscanInner(isCore, neighborhoods, m.Labels)
	return m

}

// GetNOutputs returns output columns number for Y to pass to predict
func (m *DBSCAN) GetNOutputs() int { return 1 }

// Predict for DBSCAN return Labels in Y. X must me the same passed to Fit
func (m *DBSCAN) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, m.GetNOutputs(), nil)
	}
	// return m.Labels in Y
	ySamples, yCols := Y.Dims()
	if nSamples != len(m.Labels) || ySamples != len(m.Labels) || yCols != 1 {
		panic("X must me the same passed to Fit and Y must have size samples*1")
	}
	for i, label := range m.Labels {

		Y.Set(i, 0, float64(label))
	}
	return base.FromDense(Ymutable, Y)
}

// Score for DBSCAN returns 1
func (m *DBSCAN) Score(X, Y mat.Matrix) float64 { return 1 }

func dbscanInner(isCore []bool, neighborhoods [][]int, labels []int) {
	var labelNum, v, lstack int
	var neighb []int
	var stack []int
	for i := range labels {
		if labels[i] != -1 || !isCore[i] {
			continue
		}

		// # Depth-first search starting from i, ending at the non-core points.
		// # This is very similar to the classic algorithm for computing connected
		// # components, the difference being that we label non-core points as
		// # part of a cluster (component), but don't expand their neighborhoods.
		for {
			if labels[i] == -1 {
				labels[i] = labelNum
				if isCore[i] {
					neighb = neighborhoods[i]
					for i, v = range neighb {
						if labels[v] == -1 {
							stack = append(stack, v)
						}
					}
				}
			}
			lstack = len(stack)
			if lstack == 0 {
				break
			}
			i = stack[lstack-1]
			stack = stack[0 : lstack-1]
		}
		labelNum++
	}
}
