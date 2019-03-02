package exp

var (
	_ RowViewerXX = &GeneralXX{}
	_ ColViewerXX = &GeneralXX{}
	_ Matrix      = &VectorXX{}
	_ Mutable     = &VectorXX{}
)

// Vector32 like blas32.Vector
type Vector32 blas32Vector

func (v *Vector32) Len() int           { return (len(v.Data) - 1) / v.Inc }
func (v *Vector32) Less(i, j int) bool { return v.Data[i*v.Inc] < v.Data[j*v.Inc] }
func (v *Vector32) Swap(i, j int)      { v.Data[i*v.Inc], v.Data[j*v.Inc] = v.Data[j*v.Inc], v.Data[i*v.Inc] }

// AtVec returns values as pos i
func (v *Vector32) AtVec(i int) float64 { return float64(v.Data[i*v.Inc]) }

// SetVec sets value at pos i
func (v *Vector32) SetVec(i int, val float64) { v.Data[i*v.Inc] = float32(val) }

// Dims return Len(),1
func (v *Vector32) Dims() (r, c int) { return (len(v.Data) - 1) / v.Inc, 1 }

// At return value at pos i. panics if j!=0
func (v *Vector32) At(i, j int) float64 {
	if j != 0 {
		panic("err shape")
	}
	return float64(v.Data[i*v.Inc])
}

// SetAt sets Value at pos i. panics if j!=0
func (v *Vector32) SetAt(i, j int, val float64) {
	if j != 0 {
		panic("err shape")
	}
	v.Data[i*v.Inc] = float32(val)
}

// T return row-matrix. panics if Inc!=1
func (v *Vector32) T() Matrix {
	if v.Inc != 1 {
		panic("unimplemented")
	}
	return &General32{Rows: 1, Cols: len(v.Data), Stride: len(v.Data)}
}

// Vector64 like blas64.Vector
type Vector64 blas64Vector

func (v *Vector64) Len() int           { return (len(v.Data) - 1) / v.Inc }
func (v *Vector64) Less(i, j int) bool { return v.Data[i*v.Inc] < v.Data[j*v.Inc] }
func (v *Vector64) Swap(i, j int)      { v.Data[i*v.Inc], v.Data[j*v.Inc] = v.Data[j*v.Inc], v.Data[i*v.Inc] }

// AtVec returns values as pos i
func (v *Vector64) AtVec(i int) float64 { return float64(v.Data[i*v.Inc]) }

// SetVec sets value at pos i
func (v *Vector64) SetVec(i int, val float64) { v.Data[i*v.Inc] = float64(val) }

// Dims return Len(),1
func (v *Vector64) Dims() (r, c int) { return (len(v.Data) - 1) / v.Inc, 1 }

// At return value at pos i. panics if j!=0
func (v *Vector64) At(i, j int) float64 {
	if j != 0 {
		panic("err shape")
	}
	return float64(v.Data[i*v.Inc])
}

// SetAt sets Value at pos i. panics if j!=0
func (v *Vector64) SetAt(i, j int, val float64) {
	if j != 0 {
		panic("err shape")
	}
	v.Data[i*v.Inc] = float64(val)
}

// T return row-matrix. panics if Inc!=1
func (v *Vector64) T() Matrix {
	if v.Inc != 1 {
		panic("unimplemented")
	}
	return &General64{Rows: 1, Cols: len(v.Data), Stride: len(v.Data)}
}

// VectorXX like blasXX.Vector
type VectorXX = Vector32

// Vector like blas64.Vector
type Vector = Vector64

// RowViewer32 returns row as a vector
type RowViewer32 interface {
	RowView(i int) Vector32
}

// RowViewer64  returns row as a vector
type RowViewer64 interface {
	RowView(i int) Vector64
}

// RowViewerXX returns row as a vector
type RowViewerXX = RowViewer32

// RowViewer returns row as a vector
type RowViewer = RowViewer64

// ColViewer32 returns Col as a vector
type ColViewer32 interface {
	ColView(i int) Vector32
}

// ColViewer64  returns Col as a vector
type ColViewer64 interface {
	ColView(i int) Vector64
}

// ColViewerXX returns Col as a vector
type ColViewerXX = ColViewer32

// ColViewer returns Col as a vector
type ColViewer = ColViewer64

// RowView returns row as a vector
func (mat General32) RowView(i int) Vector32 {
	return Vector32{Inc: 1, Data: mat.Data[i*mat.Stride : i*mat.Stride+mat.Cols]}
}

// RowView returns row as a vector
func (mat General64) RowView(i int) Vector {
	return Vector{Inc: 1, Data: mat.Data[i*mat.Stride : i*mat.Stride+mat.Cols]}
}

// ColView returns Col as a vector
func (mat General32) ColView(i int) Vector32 {
	return Vector32{Inc: mat.Stride, Data: mat.Data[i : (mat.Rows-1)*mat.Stride+1]}
}

// ColView returns Col as a vector
func (mat General64) ColView(i int) Vector {
	return Vector{Inc: mat.Stride, Data: mat.Data[i : (mat.Rows-1)*mat.Stride+1]}
}
