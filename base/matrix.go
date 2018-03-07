package base

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// MatTranspose is a matrix override to transpose a mat.Matrix from its initializer
type MatTranspose struct{ mat.Matrix }

// Dims for MatTranspose
func (m MatTranspose) Dims() (int, int) { r, c := m.Matrix.Dims(); return c, r }

// At for MatTranspose
func (m MatTranspose) At(i, j int) float64 {
	return m.Matrix.At(j, i)
}

// T for MatTranspose
func (m MatTranspose) T() mat.Matrix { return m.Matrix }

// MatOnesPrepended is a matrix override representing its initializer with an initial column of ones added
type MatOnesPrepended struct{ mat.Matrix }

// Dims for MatOnesPrepended
func (m MatOnesPrepended) Dims() (int, int) { r, c := m.Matrix.Dims(); return r, 1 + c }

// At for MatOnesPrepended
func (m MatOnesPrepended) At(i, j int) float64 {
	if j == 0 {
		return 1.
	}
	return m.Matrix.At(i, j-1)
}

// T for MatOnesPrepended not implemented
func (m MatOnesPrepended) T() mat.Matrix { return MatTranspose{m} }

// MatFirstColumnRemoved is a matrix whose an initial column has been removed respective to its initializer
type MatFirstColumnRemoved struct{ mat.Matrix }

// Dims for MatFirstColumnRemoved
func (m MatFirstColumnRemoved) Dims() (int, int) { r, c := m.Matrix.Dims(); return r, c - 1 }

// At for MatFirstColumnRemoved
func (m MatFirstColumnRemoved) At(i, j int) float64 {
	return m.Matrix.At(i, j+1)
}

// T for MatFirstColumnRemoved not implemented
func (m MatFirstColumnRemoved) T() mat.Matrix { panic("MatFirstColumnRemoved T not implemented") }

// MatApply0 is a mat.Matrix override where At returns a func-generated value
type MatApply0 struct {
	Rows, Columns int
	Func          func() float64
}

// Dims for MatApply0
func (m MatApply0) Dims() (int, int) { return m.Rows, m.Columns }

// At for MatApply0
func (m MatApply0) At(i, j int) float64 { return m.Func() }

// T for MatApply0
func (m MatApply0) T() mat.Matrix { return MatTranspose{m} }

// MatApply1 is a mat.Matrix override where At returns a func-trancformed value whose inputs are elements from its initializers
type MatApply1 struct {
	mat.Matrix
	Func func(float64) float64
}

// Dims for MatApply1
func (m MatApply1) Dims() (int, int) { return m.Matrix.Dims() }

// At for MatApply1
func (m MatApply1) At(i, j int) float64 { return m.Func(m.Matrix.At(i, j)) }

// T for MatApply1 returns a MatTranspose
func (m MatApply1) T() mat.Matrix { return MatTranspose{m} }

// MatApply2 is a mat.Matric overrides returning a function where args are elements from its two Matrix initializers
type MatApply2 struct {
	A, B mat.Matrix
	Func func(a, b float64) float64
}

// Dims for MatApply2
func (m MatApply2) Dims() (int, int) { return m.A.Dims() }

// At for MatApply2
func (m MatApply2) At(i, j int) float64 { return m.Func(m.A.At(i, j), m.B.At(i, j)) }

// T for MatApply2
func (m MatApply2) T() mat.Matrix { return MatTranspose{m} }

// MatSub is a mat.Matrix override returning difference from its two initializers
type MatSub struct{ A, B mat.Matrix }

// Dims for MatSub
func (m MatSub) Dims() (int, int) { return m.A.Dims() }

// At for MatSub
func (m MatSub) At(i, j int) float64 { return m.A.At(i, j) - m.B.At(i, j) }

// T for MatSub
func (m MatSub) T() mat.Matrix { return MatTranspose{m} }

// MatMulElem is a mat.Matrix override returning elementwize product from its two initializers
type MatMulElem struct{ A, B mat.Matrix }

// Dims for MatMuilElem
func (m MatMulElem) Dims() (int, int) { return m.A.Dims() }

// At for MatMulElem
func (m MatMulElem) At(i, j int) float64 { return m.A.At(i, j) * m.B.At(i, j) }

// T for MatMulElem
func (m MatMulElem) T() mat.Matrix { return MatTranspose{m} }

// MatScaled  is a mat.Matrix override returning scaled value from its initializer
type MatScaled struct {
	mat.Matrix
	float64
}

// Dims for MatScaled
func (m MatScaled) Dims() (int, int) { return m.Matrix.Dims() }

// At for MatScaled
func (m MatScaled) At(i, j int) float64 { return m.Matrix.At(i, j) * m.float64 }

// T for MatScaled
func (m MatScaled) T() mat.Matrix { return MatTranspose{m} }

// MatOneMinus  is a mat.Matrix override returning 1.-value from its initializer
type MatOneMinus struct {
	mat.Matrix
}

// Dims for MatOnesMinus
func (m MatOneMinus) Dims() (int, int) { return m.Dims() }

// At for MatOneMinus
func (m MatOneMinus) At(i, j int) float64 { return 1. - m.At(i, j) }

// T for MatOneMinus
func (m MatOneMinus) T() mat.Matrix { return MatTranspose{m} }

// MatDimsString returns a string representing Dims of its several Matrix parameters
func MatDimsString(mats ...mat.Matrix) string {
	s := ""
	for _, m := range mats {
		r, c := m.Dims()
		s = fmt.Sprintf("%s %d,%d", s, r, c)
	}
	return s
}

// MatDimsCheck checks compat of operator op and its Matrix parameters Dims.
// R is result of op, X and Y are operands of op.
// "." is dot product. "+","-","*","/" are elementwize opts
func MatDimsCheck(op string, R, X, Y mat.Matrix) {
	rx, cx := X.Dims()
	ry, cy := Y.Dims()
	rr, cr := R.Dims()
	switch op {
	case "+", "-", "*", "/":
		if rx != ry || cx != cy || rr != rx || cr != cx {
			panic(fmt.Errorf("%s %s", op, MatDimsString(R, X, Y)))
		}
	case ".":
		if cx != ry || rr != rx || cr != cy {
			panic(fmt.Errorf("%s %s", op, MatDimsString(R, X, Y)))
		}
	}
}
