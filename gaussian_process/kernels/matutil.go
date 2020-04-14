package kernels

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	//"gorgonia.org/tensor"
	"github.com/pa-m/sklearn/gaussian_process/tensor"
	"math"
)

// NewMatFromFunc ...
func NewMatFromFunc(r, c int, at func(i, j int) float64, set func(i, j int, v float64)) mat.Matrix {
	return &matFromFunc{r: r, c: c, at: at, set: set}
}

type matFromFunc struct {
	r, c int
	at   func(i, j int) float64
	set  func(i, j int, v float64)
}
type transposed struct {
	mat.Matrix
}

func (m transposed) Dims() (r, c int) {
	c, r = m.Matrix.Dims()
	return
}
func (m transposed) At(i, j int) float64 {
	return m.Matrix.At(j, i)
}

func (m transposed) T() mat.Matrix {
	return m.Matrix
}

func (m matFromFunc) Dims() (r, c int) {
	return m.r, m.c
}
func (m matFromFunc) At(i, j int) float64 {
	return m.at(i, j)
}
func (m matFromFunc) Set(i, j int, v float64) {
	m.set(i, j, v)
}

func (m matFromFunc) T() mat.Matrix {
	return transposed{Matrix: m}
}

func matCopy(dst mat.Mutable, src mat.Matrix) {
	r, c := src.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dst.Set(i, j, src.At(i, j))
		}
	}
}

func mulElemSym(dstsym, asym, bsym *mat.SymDense) {
	dst, a, b := dstsym.RawSymmetric(), asym.RawSymmetric(), bsym.RawSymmetric()
	for i, pd, pa, pb := 0, 0, 0, 0; i < a.N; i, pd, pa, pb = i+1, pd+dst.Stride, pa+a.Stride, pb+b.Stride {
		for j := i; j < a.N; j++ {
			dst.Data[pd+j] = a.Data[pa+j] * b.Data[pb+j]
		}
	}
}

func applySym(dstsym, asym *mat.SymDense, f func(i, j int, v float64) float64) {
	dst, a := dstsym.RawSymmetric(), asym.RawSymmetric()
	for i, pd, pa := 0, 0, 0; i < dst.N; i, pd, pa = i+1, pd+dst.Stride, pa+a.Stride {
		for j := i; j < dst.N; j++ {
			dst.Data[pd+j] = f(i, j, a.Data[pa+j])
		}
	}
}

// Diff3D returns an error if input tensors max abs difference exceeds tol
func Diff3D(a, b tensor.Tensor, tol float64) error {
	var delta struct {
		i0, i1, i2 int
		a, b, max  float64
	}
	s := a.Shape()
	for i0 := 0; i0 < s[0]; i0++ {
		for i1 := 0; i1 < s[1]; i1++ {
			for i2 := 0; i2 < s[2]; i2++ {
				va, err := a.At(i0, i1, i2)
				if err != nil {
					panic(err)
				}
				vb, err := b.At(i0, i1, i2)
				if err != nil {
					panic(err)
				}
				d := math.Abs(va.(float64) - vb.(float64))
				if d > tol && d > delta.max {
					delta.i0 = i0
					delta.i1 = i1
					delta.i2 = i2
					delta.a = va.(float64)
					delta.b = vb.(float64)
					delta.max = d
				}
			}
		}
	}
	if delta.max > tol {
		return fmt.Errorf("not equal. at [%d,%d,%d] a=%g b=%g delta=%g", delta.i0, delta.i1, delta.i2, delta.a, delta.b, delta.max)
	}
	return nil
}
