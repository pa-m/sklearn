package preprocessing

import (
	"fmt"
	"math"

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

	// additional example from http://scikit-learn.org/stable/modules/preprocessing.html#custom-transformers
	transformer := NewFunctionTransformer(
		func(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
			Xmat := X.RawMatrix()
			X1 = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
			X1.Apply(func(i, j int, v float64) float64 { return math.Log1p(v) }, X)
			Y1 = Y
			return

		}, func(X, Y *mat.Dense) (X1, Y1 *mat.Dense) {
			Xmat := X.RawMatrix()
			X1 = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
			X1.Apply(func(i, j int, v float64) float64 { return math.Exp(v) - 1 }, X)
			Y1 = Y
			return

		},
	)
	X = mat.NewDense(2, 2, []float64{0, 1, 2, 3})
	X1, _ = transformer.Transform(X, nil)
	fmt.Printf("log1p:\n%.8f\n", mat.Formatted(X1))
	// Output:
	// ⎡2  3  4⎤
	// ⎣6  7  8⎦
	// ⎡1  2  3  4⎤
	// ⎣5  6  7  8⎦
	// log1p:
	// ⎡0.00000000  0.69314718⎤
	// ⎣1.09861229  1.38629436⎦

}
