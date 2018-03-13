package neuralNetwork

import (
	"math"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/blas/blas64"

	"gonum.org/v1/gonum/mat"
)

type matx struct{ Dense *mat.Dense }

func (m matx) RemoveFirstRow() matx {
	r, c := m.Dense.Dims()
	return matx{Dense: base.MatDenseSlice(m.Dense, 1, r, 0, c)}
}

// AddScaledApplied adds scale*B to m
func (m matx) AddScaled(scale float64, B mat.RawMatrixer) {
	amat, bmat := m.RawMatrix(), B.RawMatrix()
	for ja, jb, jm := 0, 0, 0; ja < amat.Rows*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+amat.Stride {
		for i, v := range amat.Data[ja : ja+amat.Cols] {
			amat.Data[i+jm] = v + scale*bmat.Data[i+jb]
		}
	}
}

// CopyPrependOnes copy B to m with a column of ones prepended
func (m matx) CopyPrependOnes(B mat.RawMatrixer) {
	amat, bmat := m.RawMatrix(), B.RawMatrix()
	for ja, jb, jm := 0, 0, 0; ja < amat.Rows*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+amat.Stride {
		for i := range amat.Data[ja : ja+amat.Cols] {
			if i > 0 {
				amat.Data[i+jm] = bmat.Data[i+jb-1]
			} else {
				amat.Data[i+jm] = 1.
			}
		}
	}
}

// AddScaledApplied adds scale*f(B) to m
func (m matx) AddScaledApplied(scale float64, B mat.RawMatrixer, f func(float64) float64) {
	amat, bmat := m.RawMatrix(), B.RawMatrix()
	for ja, jb, jm := 0, 0, 0; ja < amat.Rows*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+amat.Stride {
		for i, v := range amat.Data[ja : ja+amat.Cols] {
			amat.Data[i+jm] = v + scale*f(bmat.Data[i+jb])
		}
	}
}

// CopyApplied  copy f(B) to m
func (m matx) CopyApplied(B mat.RawMatrixer, f func(float64) float64) {
	amat, bmat := m.RawMatrix(), B.RawMatrix()
	for ja, jb, jm := 0, 0, 0; ja < amat.Rows*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+amat.Stride {
		for i := range amat.Data[ja : ja+amat.Cols] {
			amat.Data[i+jm] = f(bmat.Data[i+jb])
		}
	}
}

func (m matx) SumSquares() float64 {
	amat := m.RawMatrix()
	sum := 0.
	for ja := 0; ja < amat.Rows*amat.Stride; ja = ja + amat.Stride {
		for _, v := range amat.Data[ja : ja+amat.Cols] {
			sum += v * v
		}
	}
	return sum
}
func (m matx) SumAbs() float64 {
	amat := m.RawMatrix()
	sum := 0.
	for ja := 0; ja < amat.Rows*amat.Stride; ja = ja + amat.Stride {
		for _, v := range amat.Data[ja : ja+amat.Cols] {
			sum += math.Abs(v)
		}
	}
	return sum
}
func (m matx) RawMatrix() blas64.General {
	return m.Dense.RawMatrix()
}
