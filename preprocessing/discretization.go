package preprocessing

import (
	"fmt"
	"math"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// KBinsDiscretizer structure
// Encode = "onehot-dense","ordinal"
// Strategy = "quantile","uniform","kmeans"
type KBinsDiscretizer struct {
	NBins    int
	Encode   string
	Strategy string
	BinEdges [][]float64
}

// NewKBinsDiscretizer returns a discretizer with Encode="onehot-dense" ans strategy="quantile"
func NewKBinsDiscretizer(NBins int) *KBinsDiscretizer {
	return &KBinsDiscretizer{NBins: NBins, Encode: "onehot-dense", Strategy: "quantile"}
}

// TransformerClone ...
func (m *KBinsDiscretizer) TransformerClone() Transformer {
	clone := *m
	return &clone
}

// Fit fits the transformer
func (m *KBinsDiscretizer) Fit(X, Y mat.Matrix) base.Fiter {
	NSamples, NFeatures := X.Dims()
	m.BinEdges = make([][]float64, NFeatures)
	base.Parallelize(-1, NFeatures, func(th, start, end int) {
		tmp := make([]float64, NSamples)
		for f := start; f < end; f++ {
			mat.Col(tmp, f, X)
			min, max := floats.Min(tmp), floats.Max(tmp)
			m.BinEdges[f] = make([]float64, m.NBins+1)
			for b := 0; b <= m.NBins; b++ {
				switch m.Strategy {
				case "quantile":
					m.BinEdges[f][b] = stat.Quantile(float64(b)/float64(m.NBins), stat.Empirical, tmp, nil)
				case "uniform":
					m.BinEdges[f][b] = min + float64(b)/float64(m.NBins)*(max-min)
				default:
					panic(fmt.Errorf("not implemented strategy %s", m.Strategy))
				}
			}
		}
	})

	return m
}

// Transform discretizes the Data
func (m *KBinsDiscretizer) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	switch m.Encode {
	case "ordinal":
		Xout = mat.NewDense(NSamples, NFeatures, nil)
	case "onehot":
	case "onehot-dense":
		Xout = mat.NewDense(NSamples, NFeatures*m.NBins, nil)
	}
	base.Parallelize(-1, NFeatures, func(th, start, end int) {
		for f := start; f < end; f++ {
			for i := 0; i < NSamples; i++ {
				x := X.At(i, f)
				ith := 0
				for ith < m.NBins-1 {
					if m.BinEdges[f][ith+1] <= x {
						ith++
					} else {
						break
					}
				}
				switch m.Encode {
				case "ordinal":
					Xout.Set(i, f, float64(ith))
				case "onehot":
				case "onehot-dense":
					Xout.Set(i, f*m.NBins+ith, 1)
				}
			}
		}
	})

	return
}

// FitTransform fitts the data then transforms it
func (m *KBinsDiscretizer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform transforms discretized data back to original feature space.
func (m *KBinsDiscretizer) InverseTransform(X mat.Matrix, Y mat.Mutable) (Xout, Yout *mat.Dense) {
	NSamples, _ := X.Dims()
	NFeatures := len(m.BinEdges)
	Xout = mat.NewDense(NSamples, NFeatures, nil)
	base.Parallelize(-1, NFeatures, func(th, start, end int) {
		tmp := make([]float64, NSamples)
		for f := start; f < end; f++ {
			for i := 0; i < NSamples; i++ {
				var ith int
				switch m.Encode {
				case "ordinal":
					ith = int(math.Floor(X.At(i, f)))
				case "onehot", "onehot-dense":
					ith = 0
					for c := f*m.NBins + 1; c < (f+1)*m.NBins; c++ {
						if X.At(i, c) > X.At(i, f*m.NBins+ith) {
							ith = c - f*m.NBins
						}
					}
				}
				if ith < 0 {
					ith = 0
				} else if ith > m.NBins-1 {
					ith = m.NBins - 1
				}
				tmp[i] = .5 * (m.BinEdges[f][ith] + m.BinEdges[f][ith+1])
			}
			Xout.SetCol(f, tmp)
		}
	})
	return
}
