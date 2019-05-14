package neuralnetwork

import (
	"sort"

	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	_ Matrix         = &GeneralXX{}
	_ Mutable        = &GeneralXX{}
	_ RawRowViewerXX = &GeneralXX{}
	_ sort.Interface = &GeneralXX{}
	_ Slicer         = &GeneralXX{}
)

// RawMatrixer32 provide access to blas matrix
type RawMatrixer32 interface {
	RawMatrix() blas32General
}

// RawMatrixer64 provide access to blas matrix
type RawMatrixer64 interface {
	RawMatrix() blas64General
}

// RawRowViewer32 returns row as a float slice
type RawRowViewer32 interface {
	RawRowView(i int) []float32
}

// RawRowViewer64  returns row as a float slice
type RawRowViewer64 interface {
	RawRowView(i int) []float64
}

// RawRowViewerXX returns row as a float slice
type RawRowViewerXX = RawRowViewer32

// RawRowViewer returns row as a float slice
type RawRowViewer = RawRowViewer64

// Mutable provide Set to set value at row,col
type Mutable interface {
	Set(i, j int, v float64)
	Matrix
}

// Slicer provides Slice(startRow,endRow,startCol,endCol)
type Slicer interface {
	Slice(i, j, k, l int) Matrix
}

type blas32General = blas32.General
type blas64General = blas64.General
type blasXXGeneral = blas32General

// General32 is like blas32.General
type General32 blas32.General

/*type General32 struct {
	Rows, Cols, Stride int
	Data               []float32
}*/

// General64 is like blas64.General
type General64 blas64.General

/*
type General64 struct {
	Rows, Cols, Stride int
	Data               []float64
}
*/

// GeneralXX is for easy tranposition to float32 or float64
type GeneralXX = General32
type transposed struct{ Matrix }

func (mat transposed) Dims() (r, c int) {
	c, r = mat.Matrix.Dims()
	return
}
func (mat transposed) At(i, j int) float64 {
	return mat.Matrix.At(j, i)
}
func (mat transposed) T() Matrix {
	return mat.Matrix
}

// Dims return number of rows and columns
func (mat General32) Dims() (r, c int) {
	return mat.Rows, mat.Cols
}

// At returns value at row,col
func (mat General32) At(r, c int) float64 {
	return float64(mat.Data[c+r*mat.Stride])
}

// Set set value at row,col
func (mat General32) Set(r, c int, v float64) {
	mat.Data[c+r*mat.Stride] = float32(v)
}

// T returns transposed Matrix
func (mat General32) T() Matrix {
	return transposed{mat}
}

// Dims return number of rows and columns
func (mat General64) Dims() (r, c int) {
	return mat.Rows, mat.Cols
}

// At returns value at row,col
func (mat General64) At(r, c int) float64 {
	return float64(mat.Data[c+r*mat.Stride])
}

// Set set value at row,col
func (mat General64) Set(r, c int, v float64) {
	mat.Data[c+r*mat.Stride] = float64(v)
}

// T returns transposed Matrix
func (mat General64) T() Matrix {
	return transposed{mat}
}

// RawMatrix return blas raw matrix
func (mat General32) RawMatrix() blas32General {
	return blas32General{Rows: mat.Rows, Cols: mat.Cols, Stride: mat.Stride, Data: mat.Data}
}

// RawMatrix return blas raw matrix
func (mat General64) RawMatrix() blas64General {
	return blas64General{Rows: mat.Rows, Cols: mat.Cols, Stride: mat.Stride, Data: mat.Data}
}

// RawRowView returns row as a float slice
func (mat General32) RawRowView(i int) []float32 {
	return mat.Data[i*mat.Stride : i*mat.Stride+mat.Cols]
}

// RawRowView returns row as a float slice
func (mat General64) RawRowView(i int) []float64 {
	return mat.Data[i*mat.Stride : i*mat.Stride+mat.Cols]
}

// Len returns row count
func (mat General32) Len() int {
	return mat.Rows
}

// Len returns row count
func (mat General64) Len() int {
	return mat.Rows
}

// Less compare rows. panics if Cols!=1
func (mat General32) Less(i, j int) bool {
	if mat.Cols != 1 {
		panic("cols != 1")
	}
	return mat.Data[i*mat.Stride] < mat.Data[j*mat.Stride]
}

// Less compare rows. panics if Cols!=1
func (mat General64) Less(i, j int) bool {
	if mat.Cols != 1 {
		panic("cols != 1")
	}
	return mat.Data[i*mat.Stride] < mat.Data[j*mat.Stride]
}

// Swap permutes 2 lines {
func (mat General32) Swap(i, j int) {
	for k, p, q := 0, i*mat.Stride, j*mat.Stride; k < mat.Cols; k, p, q = k+1, p+1, q+1 {
		mat.Data[p], mat.Data[q] = mat.Data[q], mat.Data[p]
	}
}

// Swap permutes 2 lines {
func (mat General64) Swap(i, j int) {
	for k, p, q := 0, i*mat.Stride, j*mat.Stride; k < mat.Cols; k, p, q = k+1, p+1, q+1 {
		mat.Data[p], mat.Data[q] = mat.Data[q], mat.Data[p]
	}
}

type general32FastSwapT struct {
	General32
	tmp []float32
}

func (g general32FastSwapT) Swap(i, j int) {
	ipos := i * g.Stride
	copy(g.tmp, g.Data[ipos:ipos+g.Cols])
	jpos := j * g.Stride
	copy(g.Data[ipos:ipos+g.Cols], g.Data[jpos:jpos+g.Cols])
	copy(g.Data[jpos:jpos+g.Cols], g.tmp)
}
func general32FastSwap(g blas32.General) general32FastSwapT {
	return general32FastSwapT{General32(g), make([]float32, g.Cols)}
}

type general64FastSwapT struct {
	General64
	tmp []float64
}

func (g general64FastSwapT) Swap(i, j int) {
	ipos := i * g.Stride
	jpos := j * g.Stride
	if g.Cols == 1 {
		g.Data[ipos], g.Data[jpos] = g.Data[jpos], g.Data[ipos]
		return
	}
	copy(g.tmp, g.Data[ipos:ipos+g.Cols])
	copy(g.Data[ipos:ipos+g.Cols], g.Data[jpos:jpos+g.Cols])
	copy(g.Data[jpos:jpos+g.Cols], g.tmp)
}
func general64FastSwap(g blas64.General) general64FastSwapT {
	return general64FastSwapT{General64(g), make([]float64, g.Cols)}
}

// Slice provides view on submatrix(startRow,endRow,startCol,endCol)
// returned matrix can be casted to *General32
func (mat General32) Slice(i, j, k, l int) Matrix {
	return General32{Rows: j - i, Cols: l - k, Stride: mat.Stride, Data: mat.Data[i*mat.Stride+k : (j-1)*mat.Stride+l]}
}

// Slice provides view on submatrix(startRow,endRow,startCol,endCol)
// returned matrix can be casted to *General64
func (mat General64) Slice(i, j, k, l int) Matrix {
	return General64{Rows: j - i, Cols: l - k, Stride: mat.Stride, Data: mat.Data[i*mat.Stride+k : (j-1)*mat.Stride+l]}
}

// RowSlice provides view on submatrix(startRow,endRow) as General32
// returned matrix can be casted to *General32
func (mat General32) RowSlice(i, j int) General32 {
	return General32{Rows: j - i, Cols: mat.Cols, Stride: mat.Stride, Data: mat.Data[i*mat.Stride : j*mat.Stride]}
}

// RowSlice provides view on submatrix(startRow,endRow) as General64
// returned matrix can be casted to *General64
func (mat General64) RowSlice(i, j int) General64 {
	return General64{Rows: j - i, Cols: mat.Cols, Stride: mat.Stride, Data: mat.Data[i*mat.Stride : j*mat.Stride]}
}

// Copy fills receiver with input matrix
func (mat *General32) Copy(a Matrix) {
	rows, cols := a.Dims()
	size := rows * cols
	if cap(mat.Data) < size {
		*mat = General32{Rows: rows, Cols: cols, Stride: cols, Data: make([]float32, size)}
	} else {
		*mat = General32{Rows: rows, Cols: cols, Stride: cols, Data: mat.Data[:size]}
	}
	if g, ok := a.(*General32); ok {
		copy(mat.Data, g.Data)
	} else if g, ok := a.(RawMatrixer32); ok && mat.Stride == g.RawMatrix().Stride {
		copy(mat.Data, g.RawMatrix().Data)
	} else if rrv, ok := a.(RawRowViewer32); ok {
		for r, rowpos := 0, 0; r < rows; r, rowpos = r+1, rowpos+mat.Stride {
			copy(mat.Data[rowpos:rowpos+mat.Cols], rrv.RawRowView(r))
		}
	} else {
		for r, rowpos := 0, 0; r < rows; r, rowpos = r+1, rowpos+mat.Stride {
			for c := 0; c < cols; c++ {
				mat.Data[rowpos+c] = float32(a.At(r, c))
			}
		}
	}
}

// Copy fills receiver with input matrix
func (mat *General64) Copy(a Matrix) {
	rows, cols := a.Dims()
	size := rows * cols
	if cap(mat.Data) < size {
		*mat = General64{Rows: rows, Cols: cols, Stride: cols, Data: make([]float64, size)}
	} else {
		*mat = General64{Rows: rows, Cols: cols, Stride: cols, Data: mat.Data[:size]}
	}
	if g, ok := a.(General64); ok {
		copy(mat.Data, g.Data)
	} else if g, ok := a.(RawMatrixer64); ok && mat.Stride == g.RawMatrix().Stride {
		copy(mat.Data, g.RawMatrix().Data)
	} else if rrv, ok := a.(RawRowViewer64); ok {
		for r, rowpos := 0, 0; r < rows; r, rowpos = r+1, rowpos+mat.Stride {
			copy(mat.Data[rowpos:rowpos+mat.Cols], rrv.RawRowView(r))
		}
	} else {
		for r, rowpos := 0, 0; r < rows; r, rowpos = r+1, rowpos+mat.Stride {
			for c := 0; c < cols; c++ {
				mat.Data[rowpos+c] = float64(a.At(r, c))
			}
		}
	}
}

// SumRows sums rows of a into mat
func (mat *General32) SumRows(a General32) {
	for i, mpos, apos := 0, 0, 0; i < a.Rows; i, mpos, apos = i+1, mpos+mat.Stride, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			mat.Data[c] = a.Data[apos+c]
		}
	}
}

// SumRows sums rows of a into mat
func (mat *General64) SumRows(a General64) {
	for i, mpos, apos := 0, 0, 0; i < a.Rows; i, mpos, apos = i+1, mpos+mat.Stride, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			mat.Data[c] = a.Data[apos+c]
		}
	}
}
