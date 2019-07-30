package kernels

import (
	"gonum.org/v1/gonum/mat"
)

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

func mulElemSym(dstsym,asym,bsym *mat.SymDense) {
	dst,a,b:=dstsym.RawSymmetric(),asym.RawSymmetric(),bsym.RawSymmetric()
	for i,pd,pa,pb:=0,0,0,0;i<a.N;i,pd,pa,pb=i+1,pd+dst.Stride,pa+a.Stride,pb+b.Stride {
		for j:=i;j<a.N;j++ {
			dst.Data[pd+j]=a.Data[pa+j]*b.Data[pb+j]
		}
	}
}

func applySym(dstsym,asym *mat.SymDense,f func(i,j int,v float64)float64){
	dst,a:=dstsym.RawSymmetric(),asym.RawSymmetric()
	for i,pd,pa:=0,0,0;i<dst.N;i,pd,pa=i+1,pd+dst.Stride,pa+a.Stride {
		for j:=i;j<dst.N;j++ {
			dst.Data[pd+j]=f(i,j,a.Data[pa+j])
		}
	}
}

