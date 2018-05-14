package neighbors

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func ExampleMinkowskiDistance() {
	a, b := mat.NewVecDense(4, []float64{0, 0, 0, 0}), mat.NewVecDense(4, []float64{1, 1, 0, 1})
	fmt.Printf("%.8f\n", MinkowskiDistance(1)(a, b))
	fmt.Printf("%.8f\n", MinkowskiDistance(2)(a, b))
	fmt.Printf("%.8f\n", MinkowskiDistance(math.Inf(1))(a, b))
	// Output:
	//3.00000000
	//1.73205081
	//1.00000000
}

func ExampleEuclideanDistance() {
	a, b := mat.NewVecDense(4, []float64{0, 0, 0, 0}), mat.NewVecDense(4, []float64{1, 1, 0, 1})
	fmt.Printf("%.8f\n", EuclideanDistance(a, b))
	// Output:
	// 1.73205081
}
