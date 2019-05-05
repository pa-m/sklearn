package preprocessing

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleKBinsDiscretizer() {
	// example from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer.fit
	X := mat.NewDense(4, 4, []float64{
		-2, 1, -4, -1,
		-1, 2, -3, -0.5,
		0, 3, -2, 0.5,
		1, 4, -1, 2})
	est := NewKBinsDiscretizer(3)
	est.Encode = "ordinal"
	est.Strategy = "uniform"
	Xt, _ := est.FitTransform(X, nil)
	fmt.Printf("Xt:\n%g\n", mat.Formatted(Xt))
	fmt.Printf("est.BinEdges[0]:\n%g\n", est.BinEdges[0])
	Xinv, _ := est.InverseTransform(Xt, nil)
	fmt.Printf("est.InverseTransform(Xt):\n%g\n", mat.Formatted(Xinv))
	// Output:
	// Xt:
	// ⎡0  0  0  0⎤
	// ⎢1  1  1  0⎥
	// ⎢2  2  2  1⎥
	// ⎣2  2  2  2⎦
	// est.BinEdges[0]:
	// [-2 -1 0 1]
	// est.InverseTransform(Xt):
	// ⎡-1.5   1.5  -3.5  -0.5⎤
	// ⎢-0.5   2.5  -2.5  -0.5⎥
	// ⎢ 0.5   3.5  -1.5   0.5⎥
	// ⎣ 0.5   3.5  -1.5   1.5⎦

}
