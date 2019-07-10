package kernels

import "gonum.org/v1/gonum/mat"

func denseVStack(b1,b2 *mat.Dense)*mat.Dense {
	r1,c1:=b1.Dims()
	r2,_:=b2.Dims()
	b0:=mat.NewDense(r1+r2,c1,nil)
	copy(b0.Slice(0,r1,0,c1).(*mat.Dense).RawMatrix().Data,b1.RawMatrix().Data)
	copy(b0.Slice(r1,r1+r2,0,c1).(*mat.Dense).RawMatrix().Data,b2.RawMatrix().Data)
	return b0
}

type matFromFunc struct {
	r,c int
	at func(i,j int)float64
	set func(i,j int,v float64)
}
type transposed struct {
	mat.Matrix
}
func (m transposed)Dims() (r,c int) {
	c,r=m.Matrix.Dims()
	return
}
func (m transposed)At(i,j int)float64 {
	return m.Matrix.At(j,i)
}

func (m transposed)T()mat.Matrix{
	return m.Matrix
}

func (m matFromFunc)Dims() (r,c int){
	return m.r,m.c
}
func (m matFromFunc)At(i,j int)float64{
	return m.at(i,j)
}
func (m matFromFunc)T()mat.Matrix {
	return transposed{Matrix:m}
}
