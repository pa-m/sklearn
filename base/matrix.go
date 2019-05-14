package base

import (
	"bytes"
	"fmt"
	"io"

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
	nSamples, _ := Xs[0].Dims()
	b := bytes.NewBuffer(nil)

	for i := 0; i < nSamples; i++ {
		for imat, X := range Xs {
			_, nFeatures := X.Dims()
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

// MatDenseSlice returns a *mat.Dense view of partial underlaying data of M
func MatDenseSlice(src mat.RawMatrixer, i, k, j, l int) *mat.Dense {
	m := &mat.Dense{}
	m.SetRawMatrix(MatGeneralSlice(src.RawMatrix(), i, k, j, l))
	return m
}

// MatGeneralSlice returns a blas64.General view of partial underlaying data of M
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

// MatDenseRowSlice returns a *mat.Dense view of partial underlaying data of M
func MatDenseRowSlice(src mat.RawMatrixer, i, k int) *mat.Dense {
	M := src.RawMatrix()
	m := &mat.Dense{}
	m.SetRawMatrix(MatGeneralRowSlice(M, i, k))
	return m
}

// MatDenseColSlice returns a *mat.Dense  view of partial underlaying data of M
func MatDenseColSlice(src mat.RawMatrixer, j, l int) *mat.Dense {
	M := src.RawMatrix()
	m := &mat.Dense{}
	m.SetRawMatrix(MatGeneralColSlice(M, j, l))
	return m
}

// MatGeneralRowSlice returns a blas64.General view of partial underlaying data of M
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

// MatGeneralColSlice returns a blas64.General view of partial underlaying data of M
func MatGeneralColSlice(M blas64.General, j, l int) blas64.General {
	if l <= j {
		panic(fmt.Errorf("l<j %d %d", j, l))
	}
	return blas64.General{
		Rows:   M.Rows,
		Cols:   l - j,
		Stride: M.Stride,
		Data:   M.Data[j : l+(M.Rows-1)*M.Stride],
	}
}

// ToDense returns w view of m if m is a RawMatrixer, et returns a dense copy of m
func ToDense(m mat.Matrix) *mat.Dense {
	if d, ok := m.(*mat.Dense); ok {
		return d
	}
	if m == mat.Matrix(nil) {
		return &mat.Dense{}
	}
	ret := &mat.Dense{}
	if rawmatrixer, ok := m.(mat.RawMatrixer); ok {
		if rawmatrixer == mat.RawMatrixer(nil) {
			return nil
		}
		ret.SetRawMatrix(rawmatrixer.RawMatrix())
	} else {
		ret = mat.DenseCopyOf(m)
	}
	return ret
}

// FromDense fills dst (mat.Mutable) with src (mat.Dense)
func FromDense(dst mat.Mutable, dense *mat.Dense) *mat.Dense {
	if dst == mat.Mutable(nil) {
		return dense
	}
	src := dense.RawMatrix()
	if rawmatrixer, ok := dst.(mat.RawMatrixer); ok {
		dstmat := rawmatrixer.RawMatrix()
		if &dstmat.Data[0] == &src.Data[0] {
			return dense
		}
		for r, srcpos, dstpos := 0, 0, 0; r < src.Rows; r, srcpos, dstpos = r+1, srcpos+src.Stride, dstpos+dstmat.Stride {
			for c := 0; c < src.Cols; c++ {
				dstmat.Data[dstpos+c] = src.Data[srcpos+c]
			}
		}
		return dense
	}
	for r, pos := 0, 0; r < src.Rows; r, pos = r+1, pos+src.Stride {
		for c := 0; c < src.Cols; c++ {
			dst.Set(r, c, src.Data[pos+c])
		}
	}
	return dense
}
