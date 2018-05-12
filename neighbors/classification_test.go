package neighbors

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleKNeighborsClassifier() {
	X := mat.NewDense(4, 1, []float64{0, 1, 2, 3})
	Y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	neigh := NewKNeighborsClassifier(3, "uniform")
	neigh.Fit(X, Y)
	Xtest := mat.NewDense(1, 1, []float64{1.1})
	Ypred := mat.NewDense(1, 1, nil)
	neigh.Predict(Xtest, Ypred)
	fmt.Println(mat.Formatted(Ypred))
	Xtest.Set(0, 0, 0.9)

	Yprob := mat.NewDense(1, 2, nil)
	neigh.PredictProba(Xtest, Yprob)
	fmt.Printf("%.8f\n", mat.Formatted(Yprob))

	// Output:
	// [0]
	// [0.66666667  0.33333333]
}
