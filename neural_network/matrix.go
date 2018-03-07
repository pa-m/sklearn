package neuralNetwork

import (
	"gonum.org/v1/gonum/mat"
)

// matTranspose is a matrix transposed from its initializer
type matTranspose struct{ mat.Matrix }

// Dims for matTranspose
func (m matTranspose) Dims() (int, int) { r, c := m.Matrix.Dims(); return c, r }

// At for matTranspose
func (m matTranspose) At(i, j int) float64 {
	return m.Matrix.At(j, i)
}

// T for matTranspose
func (m matTranspose) T() mat.Matrix { return m.Matrix }

type matApply struct {
	mat.Matrix
	f func(float64) float64
}

func (m matApply) Dims() (int, int)    { return m.Matrix.Dims() }
func (m matApply) At(i, j int) float64 { return m.f(m.Matrix.At(i, j)) }
func (m matApply) T() mat.Matrix       { return matTranspose{m} }

// onesAddedMat is a matrix with an initial column of ones has been added
type onesAddedMat struct{ mat.Matrix }

// Dims for onesAddedMat
func (m onesAddedMat) Dims() (int, int) { r, c := m.Matrix.Dims(); return r, 1 + c }

// At for onesAddedMat
func (m onesAddedMat) At(i, j int) float64 {
	if j == 0 {
		return 1.
	}
	return m.Matrix.At(i, j-1)
}

// T for onesAddedMat not implemented
func (m onesAddedMat) T() mat.Matrix { return matTranspose{m} }

// firstColumnRemovedMat is a matrix whose an initial column has been removed respective to its initializer
type firstColumnRemovedMat struct{ mat.Matrix }

// Dims for firstColumnRemovedMat
func (m firstColumnRemovedMat) Dims() (int, int) { r, c := m.Matrix.Dims(); return r, c - 1 }

// At for firstColumnRemovedMat
func (m firstColumnRemovedMat) At(i, j int) float64 {
	return m.Matrix.At(i, j+1)
}

// T for firstColumnRemovedMat not implemented
func (m firstColumnRemovedMat) T() mat.Matrix { panic("firstColumnRemovedMat T not implemented") }

type matApply0 struct {
	r, c int
	f    func() float64
}

func (m matApply0) Dims() (int, int)    { return m.r, m.c }
func (m matApply0) At(i, j int) float64 { return m.f() }
func (m matApply0) T() mat.Matrix       { return matTranspose{m} }

type matApply2 struct {
	A, B mat.Matrix
	f    func(a, b float64) float64
}

func (m matApply2) Dims() (int, int)    { return m.A.Dims() }
func (m matApply2) At(i, j int) float64 { return m.f(m.A.At(i, j), m.B.At(i, j)) }
func (m matApply2) T() mat.Matrix       { return matTranspose{m} }

type matSub struct{ A, B mat.Matrix }

func (m matSub) Dims() (int, int)    { return m.A.Dims() }
func (m matSub) At(i, j int) float64 { return m.A.At(i, j) - m.B.At(i, j) }
func (m matSub) T() mat.Matrix       { return matTranspose{m} }

type matMulElem struct{ A, B mat.Matrix }

func (m matMulElem) Dims() (int, int)    { return m.A.Dims() }
func (m matMulElem) At(i, j int) float64 { return m.A.At(i, j) * m.B.At(i, j) }
func (m matMulElem) T() mat.Matrix       { return matTranspose{m} }

type matScale struct {
	mat.Matrix
	float64
}

func (m matScale) Dims() (int, int)    { return m.Matrix.Dims() }
func (m matScale) At(i, j int) float64 { return m.Matrix.At(i, j) * m.float64 }
func (m matScale) T() mat.Matrix       { return matTranspose{m} }

type matOneMinus struct {
	mat.Matrix
}

func (m matOneMinus) Dims() (int, int)    { return m.Dims() }
func (m matOneMinus) At(i, j int) float64 { return 1. - m.At(i, j) }
func (m matOneMinus) T() mat.Matrix       { return matTranspose{m} }
