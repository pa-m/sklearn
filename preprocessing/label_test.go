package preprocessing

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleLabelBinarizer() {
	X, Y := (*mat.Dense)(nil), mat.NewDense(5, 1, []float64{1, 2, 6, 4, 2})
	lb := &LabelBinarizer{}
	lb.Fit(X, Y)
	fmt.Println(lb.Classes)

	_, Yout := lb.Transform(nil, mat.NewDense(2, 1, []float64{1, 6}))
	fmt.Println(mat.Formatted(Yout))
	_, Y2 := lb.InverseTransform(nil, Yout)
	fmt.Println(mat.Formatted(Y2.T()))
	// Output:
	// [[1 2 4 6]]
	// ⎡1  0  0  0⎤
	// ⎣0  0  0  1⎦
	// [1  6]
}

func ExampleMultiLabelBinarizer() {

	mlb := NewMultiLabelBinarizer()

	fmt.Println("NewMultiLabelBinarizer matrix test")
	Y0 := mat.NewDense(2, 2, []float64{1, 3, 2, 3})
	_, Y1 := mlb.FitTransform(nil, Y0)
	fmt.Println(mat.Formatted(Y1))
	fmt.Println("Classes", mlb.Classes)
	_, Y2 := mlb.InverseTransform(nil, Y1)
	fmt.Println(mat.Formatted(Y2.(*mat.Dense)))

	fmt.Println("NewMultiLabelBinarizer string test")
	_, Y1 = mlb.FitTransform2(nil, [][]string{{"sci-fi", "thriller"}, {"comedy", "comedy"}})
	fmt.Println(mat.Formatted(Y1))
	fmt.Println("Classes", mlb.Classes)
	_, Y2s := mlb.InverseTransform(nil, Y1)
	fmt.Println(Y2s)
	// Output:
	// NewMultiLabelBinarizer matrix test
	// ⎡1  0  0  0  0  1⎤
	// ⎣0  1  0  0  0  1⎦
	// Classes [1 2 3]
	// ⎡1  3⎤
	// ⎣2  3⎦
	// NewMultiLabelBinarizer string test
	// ⎡0  1  0  0  0  1⎤
	// ⎣1  0  0  1  0  0⎦
	// Classes [comedy sci-fi thriller]
	// [[sci-fi thriller] [comedy comedy]]

}

func ExampleLabelEncoder() {
	// adapted from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
	le := NewLabelEncoder()
	Y := mat.NewDense(4, 1, []float64{1, 2, 2, 6})
	le.Fit(nil, Y)
	fmt.Println(le.Classes)
	_, Y1 := le.Transform(nil, mat.NewDense(4, 1, []float64{1, 1, 2, 6}))
	fmt.Println(mat.Formatted(Y1.T()))
	_, Y2 := le.InverseTransform(nil, mat.NewDense(4, 1, []float64{0, 0, 1, 2}))
	fmt.Println(mat.Formatted(Y2.T()))
	// Output:
	// [[1 2 6]]
	// [0  0  1  2]
	// [1  1  2  6]

}
