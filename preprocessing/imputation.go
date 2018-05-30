package preprocessing

import (
	"math"
	"sort"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Inputer ...
// Stragegy is mean|median|most_frequent. default to mean
type Inputer struct{ Strategy string }

// NewInputer ...
func NewInputer() *Inputer { return &Inputer{} }

// Fit for Inputer ...
func (m *Inputer) Fit(X, Y *mat.Dense) base.Transformer { return m }

// Transform for Inputer ...
func (m *Inputer) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xmat := X.RawMatrix()
	Xout, Yout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil), Y
	Xmat, Xoutmat := X.RawMatrix(), Xout.RawMatrix()
	base.Parallelize(-1, Xmat.Cols, func(th, start, end int) {
		tmp := make([]float64, Xmat.Rows, Xmat.Rows)
		var def, v float64
		for i := start; i < end; i++ {
			tmp = tmp[0:0]
			for jX := 0; jX < Xmat.Rows*Xmat.Stride; jX += Xmat.Stride {
				v = Xmat.Data[jX+i]
				if !math.IsNaN(v) {
					tmp = append(tmp, v)
				}
			}

			switch m.Strategy {
			case "median":
				sort.Float64s(tmp)
				def = stat.Quantile(.5, stat.Empirical, tmp, nil)
			case "most_frequent":
				def, _ = stat.Mode(tmp, nil)
			default:
				def = stat.Mean(tmp, nil)
			}
			for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {
				v = Xmat.Data[jX+i]
				if math.IsNaN(v) {
					v = def
				}
				Xoutmat.Data[jXout+i] = v

			}
		}

	})
	return
}

// FitTransform for Inputer ...
func (m *Inputer) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}

// InverseTransform for Inputer ...
func (m *Inputer) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = X, Y
	return
}
