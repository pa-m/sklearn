package neighbors

import (
	"fmt"
	"math"
	"time"

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

func ExampleNearestNeighbors_Tree() {
	n := 10
	X := mat.NewDense(n*n*n, 3, nil)
	for sample := 0; sample < n*n*n; sample++ {
		nf := float64(n)
		x := float64(sample)
		X.Set(sample, 0, math.Floor(x/nf/nf))
		x -= nf * nf * X.At(sample, 0)
		X.Set(sample, 1, math.Floor(x/nf))
		x -= nf * X.At(sample, 1)
		X.Set(sample, 2, x)
	}
	pts := mat.NewDense(1, 3, []float64{1.1, 5.4, 7.9})

	for _, algo := range []string{"brute", "kd_tree"} {
		neigh := NewNearestNeighbors()
		neigh.Algorithm = algo
		neigh.Fit(X)
		start := time.Now()
		distances, indices := neigh.KNeighbors(pts, 2)
		fmt.Println("elapsed", algo, time.Since(start))

		fmt.Printf("%.6f\n", mat.Formatted(distances.T()))
		fmt.Println(mat.Formatted(indices.T()))
		fmt.Println(mat.Formatted(X.RowView(int(indices.At(0, 0))).T()))
		fmt.Println(mat.Formatted(X.RowView(int(indices.At(0, 1))).T()))
	}
}
