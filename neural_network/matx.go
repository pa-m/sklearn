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

func (m matx) Orthonormalize() {
	feats, outs := m.Dense.Dims()
	if mat.Norm(m.Dense, 1) == 0. {

		return
	}
	if feats < outs {
		for i := 0; i < outs; i++ {
			row := base.MatDenseSlice(m.Dense, i, i+1, 0, outs)
			nrm := mat.Norm(row, 2)
			if nrm == 0. {
				break
			}
			row.Scale(1./nrm, row)
			for i1 := 0; i1 < i; i1++ {
				r1 := base.MatDenseSlice(m.Dense, i1, i1+1, 0, outs)
				cos := matx{}.SumApplied2(row, r1, func(r, r1 float64) float64 { return r * r1 })
				if cos == 1. {
					//fmt.Println("Orthonormalize: colin")
					continue
				}
				// splain := "\nbefore:" + base.MatStr(row)
				// splain += "\nadd   :" + base.MatStr(r1) + fmt.Sprintf("* %g", -cos)

				matx{Dense: row}.AddScaled(-cos, r1)
				// splain += "\nresult:" + base.MatStr(row)
				nrm = mat.Norm(row, 2)
				// splain += fmt.Sprintf("\nnorme2: %g\ni:%d, i1:%d\n", nrm, i, i1)
				if nrm == 0. {
					panic("can't orthonormalize")
					// return
				}
				row.Scale(1./nrm, row)
			}
		}

	} else {
		for i := 0; i < outs; i++ {
			row := base.MatDenseSlice(m.Dense, 0, feats, i, i+1)
			nrm := mat.Norm(row, 2)
			if nrm == 0. {
				break
			}
			row.Scale(1./nrm, row)
			for i1 := 0; i1 < i; i1++ {
				r1 := base.MatDenseSlice(m.Dense, 0, feats, i1, i1+1)
				cos := matx{}.SumApplied2(row, r1, func(r, r1 float64) float64 { return r * r1 })
				if cos == 1. {
					//fmt.Println("Orthonormalize: colin")
					continue
				}
				// splain := "\nbefore:" + base.MatStr(row)
				// splain += "\nadd   :" + base.MatStr(r1) + fmt.Sprintf("* %g", -cos)

				matx{Dense: row}.AddScaled(-cos, r1)
				// splain += "\nresult:" + base.MatStr(row)
				nrm = mat.Norm(row, 2)
				// splain += fmt.Sprintf("\nnorme2: %g\ni:%d, i1:%d\n", nrm, i, i1)
				if nrm == 0. {
					panic("can't orthonormalize")
					// return
				}
				row.Scale(1./nrm, row)
			}
		}
	}

}
