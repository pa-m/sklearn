package neighbors

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleNearestNeighbors() {
	// test from http://scikit-learn.org/stable/modules/neighbors.html
	X := mat.NewDense(6, 2, []float64{-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2})
	nbrs := NewNearestNeighbors()
	nbrs.Fit(X)
	distances, indices := nbrs.KNeighbors(X, 2)
	fmt.Printf("indices:\n%v\n", mat.Formatted(indices))
	fmt.Printf("distances:\n%v\n", mat.Formatted(distances))
	// Output:
	// 	indices:
	// ⎡0  1⎤
	// ⎢1  0⎥
	// ⎢2  1⎥
	// ⎢3  4⎥
	// ⎢4  3⎥
	// ⎣5  4⎦
	// distances:
	// ⎡                 0                   1⎤
	// ⎢                 0                   1⎥
	// ⎢                 0  1.4142135623730951⎥
	// ⎢                 0                   1⎥
	// ⎢                 0                   1⎥
	// ⎣                 0  1.4142135623730951⎦

}

func ExampleNearestNeighbors_KNeighborsGraph() {
	X := mat.NewDense(3, 1, []float64{0, 3, 1})
	neigh := NewNearestNeighbors()
	neigh.Fit(X)
	A := neigh.KNeighborsGraph(X, 2, "connectivity", true)
	fmt.Println(mat.Formatted(A))
	// Output:
	// ⎡1  0  1⎤
	// ⎢0  1  1⎥
	// ⎣1  0  1⎦

}
