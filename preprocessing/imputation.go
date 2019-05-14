package preprocessing

import (
	"math"
	"sort"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Imputer ...
// Stragegy is mean|median|most_frequent. default to mean
type Imputer struct {
	Strategy      string
	MissingValues []float64
}

// NewImputer ...
func NewImputer() *Imputer { return &Imputer{} }

// TransformerClone ...
func (m *Imputer) TransformerClone() base.Transformer {
	clone := *m
	return &clone
}

// Fit for Imputer ...
func (m *Imputer) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	Xmat := X.RawMatrix()
	m.MissingValues = make([]float64, Xmat.Cols)
	base.Parallelize(-1, Xmat.Cols, func(th, start, end int) {
		tmp := make([]float64, Xmat.Rows)
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
			m.MissingValues[i] = def
		}

	})
	return m
}

// Transform for Imputer ...
func (m *Imputer) Transform(Xmatrix, Ymatrix mat.Matrix) (Xout, Yout *mat.Dense) {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	Xmat := X.RawMatrix()
	Xout, Yout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil), Y
	Xmat, Xoutmat := X.RawMatrix(), Xout.RawMatrix()
	base.Parallelize(-1, Xmat.Cols, func(th, start, end int) {
		var v, def float64

		for i := start; i < end; i++ {
			def = m.MissingValues[i]
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

// FitTransform fit to dat, then transform it
func (m *Imputer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform for Imputer ...
func (m *Imputer) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = X, Y
	return
}
