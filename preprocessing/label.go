package preprocessing

import (
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// LabelBinarizer Binarize labels in a one-vs-all fashion
type LabelBinarizer struct {
	NegLabel, PosLabel float64
	Classes            [][]float64
}

// Fit for binarizer does nothing
func (m *LabelBinarizer) Fit(X, Y *mat.Dense) Transformer {
	if m.PosLabel == m.NegLabel {
		m.PosLabel += 1.
	}
	y := Y.RawMatrix()
	m.Classes = make([][]float64, y.Cols)
	for j := 0; j < y.Cols; j++ {
		cmap := make(map[float64]bool)
		for i, yi := 0, 0; i < y.Rows; i, yi = i+1, yi+y.Stride {
			yval := y.Data[yi+j]
			if _, present := cmap[yval]; present {
				continue
			}
			cmap[yval] = true
			m.Classes[j] = append(m.Classes[j], yval)
		}
		sort.Float64s(m.Classes[j])
	}
	return m
}

// Transform for LabelBinarizer
func (m *LabelBinarizer) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout = X
	NSamples, _ := Y.Dims()
	NOutputs := 0
	for _, classes := range m.Classes {
		NOutputs += len(classes)
	}

	Yout = mat.NewDense(NSamples, NOutputs, nil)
	y, yo := Y.RawMatrix(), Yout.RawMatrix()
	baseCol := 0
	for j := 0; j < y.Cols; j++ {
		cmap := make(map[float64]int)
		for classNo, val := range m.Classes[j] {
			cmap[val] = classNo
		}
		for i, yi, yo0 := 0, 0, 0; i < y.Rows; i, yi, yo0 = i+1, yi+y.Stride, yo0+yo.Stride {
			val := y.Data[yi+j]
			if classNo, ok := cmap[val]; ok {
				yo.Data[yo0+baseCol+classNo] = m.PosLabel
			} else {
				yo.Data[yo0+baseCol+classNo] = m.NegLabel
			}
		}
		baseCol += len(m.Classes[j])
	}
	return
}

// InverseTransform for LabelBinarizer
func (m *LabelBinarizer) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout = X
	NSamples, _ := Y.Dims()
	NOutputs := len(m.Classes)

	Yout = mat.NewDense(NSamples, NOutputs, nil)
	y, yo := Y.RawMatrix(), Yout.RawMatrix()
	for j, baseCol := 0, 0; baseCol < y.Cols; j, baseCol = j+1, baseCol+len(m.Classes[j]) {
		for i, yi, yo0 := 0, 0, 0; i < y.Rows; i, yi, yo0 = i+1, yi+y.Stride, yo0+yo.Stride {
			classNo := floats.MaxIdx(y.Data[yi+baseCol : yi+baseCol+len(m.Classes[j])])

			yo.Data[yo0+j] = m.Classes[j][classNo]
		}
		baseCol += len(m.Classes[j])
	}
	return
}

// FitTransform for Binarizer
func (m *LabelBinarizer) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}

// MultiLabelBinarizer Transform between iterable of iterables and a multilabel format
type MultiLabelBinarizer struct {
	Classes []interface{}

	Less func(i, j int) bool
}

// NewMultiLabelBinarizer ...
func NewMultiLabelBinarizer() *MultiLabelBinarizer { return &MultiLabelBinarizer{} }

// Fit for MultiLabelBinarizer ...
// Y type can be *mat.Dense | [][]string
func (m *MultiLabelBinarizer) Fit(X *mat.Dense, Y interface{}) *MultiLabelBinarizer {
	m.Classes = make([]interface{}, 0)
	switch vY := Y.(type) {
	case *mat.Dense:
		cmap := make(map[float64]bool)
		vYmat := vY.RawMatrix()
		for jvY := 0; jvY < vYmat.Rows*vYmat.Stride; jvY = jvY + vYmat.Stride {
			for _, v := range vYmat.Data[jvY : jvY+vYmat.Cols] {
				cmap[v] = true
			}
		}
		m.Classes = make([]interface{}, 0, len(cmap))
		for v := range cmap {
			m.Classes = append(m.Classes, v)
		}
		less := func(i, j int) bool { return m.Classes[i].(float64) < m.Classes[j].(float64) }
		sort.Slice(m.Classes, less)
	case [][]string:
		cmap := make(map[string]bool)
		for _, row := range vY {
			for _, v := range row {
				cmap[v] = true
			}
		}
		for v := range cmap {
			m.Classes = append(m.Classes, v)
		}
		less := func(i, j int) bool { return m.Classes[i].(string) < m.Classes[j].(string) }
		sort.Slice(m.Classes, less)
	default:
		panic("MultiLabelBinarizer: Y must be *mat.Dense ot [][]string")
	}
	return m
}

// Transform for MultiLabelBinarizer ...
// Y type must be the same passed int Fit
func (m *MultiLabelBinarizer) Transform(X *mat.Dense, Y interface{}) (Xout, Yout *mat.Dense) {
	Xout = X
	switch vY := Y.(type) {
	case *mat.Dense:
		Ymat := vY.RawMatrix()
		Yout = mat.NewDense(Ymat.Rows, Ymat.Cols*len(m.Classes), nil)
		Youtmat := Yout.RawMatrix()
		cmap := make(map[float64]int)
		for classNo, v := range m.Classes {
			cmap[v.(float64)] = classNo
		}
		for i, baseCol := 0, 0; i < Ymat.Cols; i, baseCol = i+1, baseCol+len(m.Classes) {
			for jY, jYout := 0, 0; jY < Ymat.Rows*Ymat.Stride; jY, jYout = jY+Ymat.Stride, jYout+Youtmat.Stride {
				v := Ymat.Data[jY+i]
				classNo, ok := cmap[v]
				if ok {
					Youtmat.Data[jYout+baseCol+classNo] = 1.
				}
			}
		}
	case [][]string:
		YRows := len(vY)

		YCols := 0
		if YRows > 0 {
			YCols = len(vY[0])
		}
		Yout = mat.NewDense(YRows, YCols*len(m.Classes), nil)
		Youtmat := Yout.RawMatrix()
		cmap := make(map[string]int)
		for classNo, v := range m.Classes {
			cmap[v.(string)] = classNo
		}
		for i, baseCol := 0, 0; i < YCols; i, baseCol = i+1, baseCol+len(m.Classes) {
			for jY, jYout := 0, 0; jY < YRows; jY, jYout = jY+1, jYout+Youtmat.Stride {
				v := vY[jY][i]
				classNo, ok := cmap[v]
				if ok {
					Youtmat.Data[jYout+baseCol+classNo] = 1.
				}
			}
		}

	}
	return
}

// FitTransform for MultiLabelBinarizer ...
// Y type can be *mat.Dense | [][]string
func (m *MultiLabelBinarizer) FitTransform(X *mat.Dense, Y interface{}) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}

// InverseTransform for MultiLabelBinarizer ...
// Yout type is same as the one passed int Fit
func (m *MultiLabelBinarizer) InverseTransform(X, Y *mat.Dense) (Xout *mat.Dense, Yout interface{}) {
	Xout = X
	switch m.Classes[0].(type) {
	case float64:
		Ymat := Y.RawMatrix()
		Yo := mat.NewDense(Ymat.Rows, Ymat.Cols/len(m.Classes), nil)
		Yomat := Yo.RawMatrix()
		for i := 0; i < Yomat.Cols; i++ {
			Ymat, Yomat := Y.RawMatrix(), Yo.RawMatrix()
			for jY, jYo := 0, 0; jY < Ymat.Rows*Ymat.Stride; jY, jYo = jY+Ymat.Stride, jYo+Yomat.Stride {
				classNo := floats.MaxIdx(Ymat.Data[jY+i*len(m.Classes) : jY+(i+1)*len(m.Classes)])
				Yomat.Data[jYo+i] = m.Classes[classNo].(float64)
			}
		}
		Yout = Yo
	case string:
		Ymat := Y.RawMatrix()
		Yo := make([][]string, Ymat.Rows)
		for j := 0; j < Ymat.Rows; j++ {
			Yo[j] = make([]string, Ymat.Cols/len(m.Classes))
		}
		for i := 0; i < len(Yo[0]); i++ {
			Ymat := Y.RawMatrix()
			for j, jY := 0, 0; j < Ymat.Rows; j, jY = j+1, jY+Ymat.Stride {
				classNo := floats.MaxIdx(Ymat.Data[jY+i*len(m.Classes) : jY+(i+1)*len(m.Classes)])
				Yo[j][i] = m.Classes[classNo].(string)
			}
		}
		Yout = Yo

	default:
		panic("MultiLabelBinarizer: unknown target type in InverseTransform")
	}
	return
}
