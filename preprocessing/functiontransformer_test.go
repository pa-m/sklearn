package preprocessing

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleFunctionTransformer() {
	X, Y := mat.NewDense(2, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8}), (*mat.Dense)(nil)

	var firstColumn []float64
	dropFirstColumn := func(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
		m, n := X.Dims()
		firstColumn = make([]float64, m)
		mat.Col(firstColumn, 0, X)
		X1 = mat.NewDense(m, n-1, nil)
		X1.Copy(X.Slice(0, m, 1, n))
		Y1 = Y
		return
	}
	undoDropFirstColumn := func(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
		m, n := X.Dims()
		n++
		X1 = mat.NewDense(m, n, nil)
		X1.SetCol(0, firstColumn)
		X1.Slice(0, m, 1, n).(*mat.Dense).Copy(X)
		Y1 = Y
		return
	}
	allButFirstColumn := NewFunctionTransformer(dropFirstColumn, undoDropFirstColumn)
	X1, _ := allButFirstColumn.Transform(X, Y)
	fmt.Println(mat.Formatted(X1))
	X2, _ := allButFirstColumn.InverseTransform(X1, nil)
	fmt.Println(mat.Formatted(X2))
	// Output:
	// ⎡2  3  4⎤
	// ⎣6  7  8⎦
	// ⎡1  2  3  4⎤
	// ⎣5  6  7  8⎦

}
