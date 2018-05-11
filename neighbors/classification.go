package neighbors

import (
	"fmt"
	"runtime"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// KNeighborsClassifier is a Regression based on k-nearest neighbors.
// The target is predicted by local interpolation of the targets
// associated of the nearest neighbors in the training set.
type KNeighborsClassifier struct {
	base.Classifier
	NearestNeighbors
	K        int
	Weight   string
	Scale    bool
	Distance Distance
	// Runtime members
	Xscaled, Y *mat.Dense
	//Classes    []map[float64]bool
}

// NewKNeighborsClassifier returns an initialized *KNeighborsClassifier
func NewKNeighborsClassifier(K int, Weights string) base.Transformer {
	return &KNeighborsClassifier{NearestNeighbors: *NewNearestNeighbors(), K: K, Weight: Weights}
}

// Fit ...
func (m *KNeighborsClassifier) Fit(X, Y *mat.Dense) base.Transformer {
	m.Xscaled = mat.DenseCopyOf(X)
	m.Y = mat.DenseCopyOf(Y)
	if m.Distance == nil {
		m.Distance = EuclideanDistance
	}
	if m.K <= 0 {
		panic(fmt.Errorf("K<=0"))
	}
	m.NearestNeighbors.Fit(X)
	// NSamples, Noutputs := Y.Dims()
	// for o := 0; o < Noutputs; o++ {
	// 	cl := make(map[float64]bool)
	// 	for s := 0; s < NSamples; s++ {
	// 		cl[Y.At(s, o)] = true
	// 	}
	// 	m.Classes = append(m.Classes,cl)
	// }
	return m
}

// Predict ...
func (m *KNeighborsClassifier) Predict(X, Y *mat.Dense) base.Transformer {
	NX, _ := X.Dims()
	_, outputs := m.Y.Dims()

	NCPU := runtime.NumCPU()
	isWeightDistance := m.Weight == "distance"
	distances, indices := m.KNeighbors(X, m.K)

	base.Parallelize(NCPU, NX, func(th, start, end int) {
		var sumd2 float64
		weights := make([]float64, m.K)
		ys := make([]float64, m.K)
		for ik := range weights {
			weights[ik] = 1.
		}
		for sample := start; sample < end; sample++ {
			// set Y(sample,output) to weighted average of K nearest
			sumd2 = floats.Sum(distances.RawRowView(sample))

			classw := make(map[float64]float64)
			for o := 0; o < outputs; o++ {
				for ik := range ys {
					cl := m.Y.At(int(indices.At(sample, ik)), o)
					if isWeightDistance {
						weights[ik] = 1. - distances.At(sample, ik)/sumd2
					}
					if clw, present := classw[cl]; present {
						classw[cl] = clw + weights[ik]
					} else {
						classw[cl] = weights[ik]
					}
				}
				wmax, clwmax := 0., 0.
				for cl, w := range classw {
					if w > wmax {
						wmax = w
						clwmax = cl
					}
				}
				Y.Set(sample, o, clwmax)
			}

		}
	})
	return m
}

// Transform for KNeighborsClassifier
func (m *KNeighborsClassifier) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, NOutputs := Y.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Yout)
	return
}

// Score for KNeighborsClassifier
func (m *KNeighborsClassifier) Score(X, Y *mat.Dense) float64 {
	NSamples, NOutputs := Y.Dims()
	Ypred := mat.NewDense(NSamples, NOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
}
