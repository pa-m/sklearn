package neighbors

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleNearestCentroid() {
	// Example adapted from from http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid
	X := mat.NewDense(6, 2, []float64{-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2})
	Y := mat.NewDense(6, 1, []float64{1, 1, 1, 2, 2, 2})
	clf := NewNearestCentroid("euclidean", 0.)
	clf.Fit(X, Y)
	Xtest := mat.NewDense(1, 2, []float64{-0.8, -1})

	Ypred := mat.NewDense(1, 1, nil)
	clf.Predict(Xtest, Ypred)
	fmt.Println(mat.Formatted(Ypred))
	// Output:
	// [1]
}
