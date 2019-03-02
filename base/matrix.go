package base

import (
	"bytes"
	"fmt"
	"golang.org/x/exp/rand"
	"io"
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// MatConst is a matrix where all cless have the same value
type MatConst struct {
	Rows, Columns int
	Value         float64
}

// Dims for MatConst
func (m MatConst) Dims() (int, int) { return m.Rows, m.Columns }

// At for MatConst
func (m MatConst) At(i, j int) float64 { return m.Value }

// T for MatConst
func (m MatConst) T() mat.Matrix { return MatTranspose{Matrix: m} }

// MatTranspose is a matrix override to transpose a mat.Matrix from its initializer
type MatTranspose struct{ mat.Matrix }

// Dims for MatTranspose
func (m MatTranspose) Dims() (int, int) { r, c := m.Matrix.Dims(); return c, r }

// At for MatTranspose
func (m MatTranspose) At(i, j int) float64 {
	return m.Matrix.At(j, i)
}

// Set for MatTranspose
func (m MatTranspose) Set(i, j int, v float64) {
	if Mutable, ok := m.Matrix.(mat.Mutable); ok {
		Mutable.Set(j, i, v)
	} else {
		panic("underling Matrix is not Mutable")
	}
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

// Set for MatOnesPrepended
func (m MatOnesPrepended) Set(i, j int, v float64) {
	if Mutable, ok := m.Matrix.(mat.Mutable); ok {
		Mutable.Set(i, j-1, v)
	} else {
		panic("underling Matrix is not Mutable")
	}
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

// Set for MatFirstColumnRemoved
func (m MatFirstColumnRemoved) Set(i, j int, v float64) {
	if Mutable, ok := m.Matrix.(mat.Mutable); ok {
		Mutable.Set(i, j+1, v)
	} else {
		panic("underling Matrix is not Mutable")
	}
}

// T for MatFirstColumnRemoved
func (m MatFirstColumnRemoved) T() mat.Matrix { return MatTranspose{m} }

// MatFirstRowZeroed is a matrix whose an initial Row has been set to zeros respective to its initializer
type MatFirstRowZeroed struct{ mat.Matrix }

// Dims for MatFirstRowZeroed
func (m MatFirstRowZeroed) Dims() (int, int) { return m.Matrix.Dims() }

// At for MatFirstRowZeroed
func (m MatFirstRowZeroed) At(i, j int) float64 {
	if i == 0 {
		return 0.
	}
	return m.Matrix.At(i, j)
}

// Set for MatFirstRowZeroed
func (m MatFirstRowZeroed) Set(i, j int, v float64) {
	if Mutable, ok := m.Matrix.(mat.Mutable); ok {
		Mutable.Set(i, j, v)
	} else {
		panic("underling Matrix is not Mutable")
	}
}

// T for MatFirstRowZeroed
func (m MatFirstRowZeroed) T() mat.Matrix { return MatTranspose{m} }

// MatRowSlice is a matrix row chunk
type MatRowSlice struct {
	mat.Matrix
	Start, End int
}

// Dims for MatRowSlice
func (m MatRowSlice) Dims() (int, int) { _, c := m.Matrix.Dims(); return m.End - m.Start, c }

// At for MatRowSlice
func (m MatRowSlice) At(i, j int) float64 {
	if i < 0 || i > m.End-m.Start {
		panic("indexing error")
	}
	return m.Matrix.At(i+m.Start, j)
}

// Set for MatRowSlice
func (m MatRowSlice) Set(i, j int, v float64) {
	if Mutable, ok := m.Matrix.(mat.Mutable); ok {
		Mutable.Set(i-m.Start, j, v)
	} else {
		panic("underling Matrix is not Mutable")
	}
}

// T for MatRowSlice
func (m MatRowSlice) T() mat.Matrix { return MatTranspose{m} }

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
	Scale float64
}

// Dims for MatScaled
func (m MatScaled) Dims() (int, int) { return m.Matrix.Dims() }

// At for MatScaled
func (m MatScaled) At(i, j int) float64 { return m.Matrix.At(i, j) * m.Scale }

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

// MatStr return a string from a mat.Matrix
func MatStr(Xs ...mat.Matrix) string {
	if len(Xs) == 0 {
		return ""
	}
	nSamples, nFeatures := Xs[0].Dims()
	b := bytes.NewBuffer(nil)

	for i := 0; i < nSamples; i++ {
		for imat, X := range Xs {
			_, nFeatures = X.Dims()
			for j := 0; j < nFeatures; j++ {
				io.WriteString(b, fmt.Sprintf("%g", X.At(i, j)))
				if j < nFeatures-1 || imat < len(Xs)-1 {
					io.WriteString(b, "\t")
				} else {
					io.WriteString(b, "\n")
				}
			}
		}
	}
	return b.String()
}

// MatColStr return the string for a matrix column
func MatColStr(X mat.Matrix, j int) string {
	nSamples, _ := X.Dims()
	var t = make([]float64, nSamples)
	mat.Col(t, j, X)
	return fmt.Sprint(t)
}

// MatRowStr returns the string for a matrix row
func MatRowStr(X mat.Matrix, i int) string {
	_, nFeatures := X.Dims()
	var t = make([]float64, nFeatures)
	mat.Row(t, i, X)
	return fmt.Sprint(t)
}

// MatShuffle shuffles the rows of X and Y matrices
func MatShuffle(X, Y *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Y.Dims()
	Xrowi := make([]float64, nFeatures, nFeatures)
	Yrowi := make([]float64, nOutputs, nOutputs)
	for i := nSamples - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		copy(Xrowi, X.RawRowView(i))
		X.SetRow(i, X.RawRowView(j))
		X.SetRow(j, Xrowi)
		copy(Yrowi, Y.RawRowView(i))
		Y.SetRow(i, Y.RawRowView(j))
		Y.SetRow(j, Yrowi)
	}
}

// MatSigmoid put emelent-wise sigmoid of X into dst
func MatSigmoid(dst *mat.Dense, X mat.Matrix) *mat.Dense {
	if dst == nil {
		r, c := X.Dims()
		dst = mat.NewDense(r, c, nil)
	}
	dst.Apply(func(i int, j int, v float64) float64 {
		return 1. / (1. + math.Exp(-v))
	}, X)
	return dst
}

// MatDenseFirstColumnRemoved returns a *mat.Dense with the same underlaying data as M but 1st column removed
func MatDenseFirstColumnRemoved(src *mat.Dense) *mat.Dense {
	nSamples, nOutputs := src.Dims()
	return MatDenseSlice(src, 0, nSamples, 1, nOutputs)
}

// MatDenseSlice returns a *mat.Dense with the same underlaying data as src but rows and columns removed
func MatDenseSlice(src mat.RawMatrixer, i, k, j, l int) *mat.Dense {
	m := &mat.Dense{}
	m.SetRawMatrix(MatGeneralSlice(src.RawMatrix(), i, k, j, l))
	return m
}

// MatGeneralSlice returns a blas64.General with the same underlaying data as M but rows and columns removed
func MatGeneralSlice(M blas64.General, i, k, j, l int) blas64.General {
	if k <= i {
		panic(fmt.Errorf("k<=i %d %d", k, i))
	}
	return blas64.General{
		Rows:   k - i,
		Cols:   l - j,
		Stride: M.Stride,
		Data:   M.Data[i*M.Stride+j : (k-1)*M.Stride+l],
	}
}

// MatDenseRowSlice returns a *mat.Dense with the same underlaying data as src but rows and columns removed
func MatDenseRowSlice(src mat.RawMatrixer, i, k int) *mat.Dense {
	M := src.RawMatrix()
	m := &mat.Dense{}
	m.SetRawMatrix(MatGeneralRowSlice(M, i, k))
	return m
}

// MatGeneralRowSlice returns a blas64.General with the same underlaying data as M but rows and columns removed
func MatGeneralRowSlice(M blas64.General, i, k int) blas64.General {
	if k <= i {
		panic(fmt.Errorf("k<=i %d %d", k, i))
	}
	return blas64.General{
		Rows:   k - i,
		Cols:   M.Cols,
		Stride: M.Stride,
		Data:   M.Data[i*M.Stride : k*M.Stride],
	}
}
