package neuralNetwork

import (
	"fmt"
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
	if amat.Rows != bmat.Rows || amat.Cols != bmat.Cols {
		panic(fmt.Errorf("%d,%d != %d,%d", amat.Rows, amat.Cols, bmat.Rows, bmat.Cols))
	}
	for ja, jb := 0, 0; ja < amat.Rows*amat.Stride; ja, jb = ja+amat.Stride, jb+bmat.Stride {
		for i, vb := range bmat.Data[jb : jb+bmat.Cols] {
			amat.Data[i+ja] = f(vb)
		}
	}
}

// CopyScaledApplied2  copy scale*f(B,C) to m
func (m matx) CopyScaledApplied2(B, C mat.RawMatrixer, scale float64, f func(float64, float64) float64) {
	amat, bmat, cmat := m.RawMatrix(), B.RawMatrix(), C.RawMatrix()
	if amat.Rows != bmat.Rows || amat.Cols != bmat.Cols {
		panic(fmt.Errorf("%d,%d != %d,%d", amat.Rows, amat.Cols, bmat.Rows, bmat.Cols))
	}
	for ja, jb, jc := 0, 0, 0; ja < amat.Rows*amat.Stride; ja, jb, jc = ja+amat.Stride, jb+bmat.Stride, jc+cmat.Stride {
		for i := range amat.Data[ja : ja+amat.Cols] {
			amat.Data[i+ja] = scale * f(bmat.Data[i+jb], cmat.Data[i+jc])
		}
	}
}

// SumApplied2  returns sum of  f(B,C) t
func (matx) SumApplied2(B, C mat.RawMatrixer, f func(float64, float64) float64) float64 {
	bmat, cmat := B.RawMatrix(), C.RawMatrix()
	if cmat.Rows != bmat.Rows || cmat.Cols != bmat.Cols {
		panic(fmt.Errorf("%d,%d != %d,%d", bmat.Rows, bmat.Cols, cmat.Rows, cmat.Cols))
	}
	sum := 0.
	for jb, jc := 0, 0; jb < bmat.Rows*bmat.Stride; jb, jc = jb+bmat.Stride, jc+cmat.Stride {
		for i := range bmat.Data[jb : jb+bmat.Cols] {
			sum += f(bmat.Data[i+jb], cmat.Data[i+jc])
		}
	}
	return sum
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
