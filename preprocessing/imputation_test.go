package preprocessing

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func ExampleInputer() {
	X := mat.NewDense(5, 2, []float64{1, 2, 3, 4, math.NaN(), 6, 7, 8, 2, 10})
	fmt.Println("replacing X.At(2,0) with...")
	for _, s := range []string{"mean", "median", "most_frequent"} {
		X1, _ := (&Inputer{Strategy: s}).Transform(X, nil)
		fmt.Printf("%s\n%g\n", s, mat.Formatted(X1))

	}
	// Output:
	// replacing X.At(2,0) with...
	// mean
	// ⎡   1     2⎤
	// ⎢   3     4⎥
	// ⎢3.25     6⎥
	// ⎢   7     8⎥
	// ⎣   2    10⎦
	// median
	// ⎡ 1   2⎤
	// ⎢ 3   4⎥
	// ⎢ 2   6⎥
	// ⎢ 7   8⎥
	// ⎣ 2  10⎦
	// most_frequent
	// ⎡ 1   2⎤
	// ⎢ 3   4⎥
	// ⎢ 7   6⎥
	// ⎢ 7   8⎥
	// ⎣ 2  10⎦
}
