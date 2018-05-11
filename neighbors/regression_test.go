package neighbors

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// ExampleKNeighborsRegressor from http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
func ExampleKNeighborsRegressor() {
	X := mat.NewDense(4, 1, []float64{0, 1, 2, 3})
	Y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	neigh := NewKNeighborsRegressor(2, "distance")
	neigh.Fit(X, Y)
	Xtest := mat.NewDense(1, 1, []float64{1.5})
	Ypred := mat.NewDense(1, 1, nil)
	neigh.Predict(Xtest, Ypred)
	fmt.Println(mat.Formatted(Ypred))
	// Output:
	// [0.5]
}
