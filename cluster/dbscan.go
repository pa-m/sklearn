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

// Fit for DBSCAN
// X : mat.Dense of shape (n_samples, n_features)
// A feature array`.
// m.SampleWeight is used if not nil
// it is the Weight of each sample, such that a sample with a weight of at least
// ``min_samples`` is by itself a core sample; a sample with negative
// weight may inhibit its eps-neighbor from being core.
// Note that weights are absolute, and default to 1.
// Y : Ignored, may be nil
func (m *DBSCAN) Fit(X, Y *mat.Dense) base.Transformer {
	m.NeighborsModel = neighbors.NewNearestNeighbors()
	m.NeighborsModel.Algorithm = m.Algorithm
	m.NeighborsModel.Distance = MinkowskiDistance(m.P)
	m.NeighborsModel.Metric = m.Metric
	m.NeighborsModel.NJobs = m.NJobs
	m.NeighborsModel.LeafSize = m.LeafSize
	m.NeighborsModel.Fit(X)
	_, neighborhoods := m.NeighborsModel.RadiusNeighbors(X, m.Eps)

	NSamples, _ := X.Dims()
	NNeighbors := make([]float64, NSamples, NSamples)

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
	m.Labels = make([]int, NSamples, NSamples)
	for i := range m.Labels {
		m.Labels[i] = -1
	}
	// # A list of all core samples found.
	isCore := make([]bool, NSamples, NSamples)
	for sample := range NNeighbors {
		if NNeighbors[sample] > m.MinSamples {
			isCore[sample] = true
			m.CoreSampleIndices = append(m.CoreSampleIndices, sample)
		}
	}
	dbscanInner(isCore, neighborhoods, m.Labels)
	return m

}

// Predict for DBSCAN
func (m *DBSCAN) Predict(X, Y *mat.Dense) base.Transformer {
	return m

}

// Transform (for pipeline)
func (m *DBSCAN) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, _ := X.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, 1, nil)
	m.Predict(X, Yout)
	return

}

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

/*def dbscan_inner(np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
                 np.ndarray[object, ndim=1] neighborhoods,
                 np.ndarray[np.npy_intp, ndim=1, mode='c'] labels):
    cdef np.npy_intp i, label_num = 0, v
    cdef np.ndarray[np.npy_intp, ndim=1] neighb
    cdef vector[np.npy_intp] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            push(stack, v)

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

label_num += 1
*/
