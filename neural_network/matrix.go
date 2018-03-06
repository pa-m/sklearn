package neuralNetwork

import (
	"gonum.org/v1/gonum/mat"
)

// transposedMat is a matrix transposed from its initializer
type transposedMat struct{ mat.Matrix }

// Dims for transposedMat
func (m transposedMat) Dims() (int, int) { r, c := m.Matrix.Dims(); return c, r }

// At for transposedMat
func (m transposedMat) At(i, j int) float64 {
	return m.Matrix.At(j, i)
}

// T for transposedMat
func (m transposedMat) T() mat.Matrix { return m.Matrix }

type appliedMat struct {
	mat.Matrix
	f func(float64) float64
}

func (m appliedMat) Dims() (int, int)    { return m.Matrix.Dims() }
func (m appliedMat) At(i, j int) float64 { return m.f(m.Matrix.At(i, j)) }
func (m appliedMat) T() mat.Matrix       { return transposedMat{m} }

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
func (m onesAddedMat) T() mat.Matrix { return transposedMat{m} }

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
