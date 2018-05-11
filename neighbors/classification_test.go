package neighbors

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleKNeighborsClassifier() {
	X := mat.NewDense(4, 1, []float64{0, 1, 2, 3})
	Y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	neigh := NewKNeighborsClassifier(2, "distance")
	neigh.Fit(X, Y)
	Xtest := mat.NewDense(1, 1, []float64{1.1})
	Ypred := mat.NewDense(1, 1, nil)
	neigh.(*KNeighborsClassifier).Predict(Xtest, Ypred)
	fmt.Println(mat.Formatted(Ypred))
	Xtest.Set(0, 0, 0.9)
	// Yprob := mat.NewDense(1, 2, nil)
	// neigh.(*KNeighborsClassifier).PredictProba(Xtest, Yprob)
	// fmt.Println(mat.Formatted(Yprob))

	// Output:
	// [0]

}
