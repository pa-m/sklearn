package preprocessing

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func ExampleImputer() {
	var nan = math.NaN()
	X := mat.NewDense(5, 2, []float64{1, 2, 3, 4, nan, 6, 7, 8, 7, 10})
	fmt.Println("replacing X.At(2,0) with...")
	for _, s := range []string{"mean", "median", "most_frequent"} {

		X1, _ := (&Imputer{Strategy: s}).FitTransform(X, nil)
		fmt.Printf("%s\n%g\n", s, mat.Formatted(X1))

	}
	// additional example adapted from http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values
	imp := NewImputer()
	imp.Fit(mat.NewDense(3, 2, []float64{1, 2, nan, 3, 7, 6}), nil)
	X = mat.NewDense(3, 2, []float64{nan, 2, 6, nan, 7, 6})
	X1, _ := imp.Transform(X, nil)
	fmt.Printf("imputation-of-missing-values:\n%g\n", mat.Formatted(X1))
	// Output:
	// replacing X.At(2,0) with...
	// mean
	// ⎡  1    2⎤
	// ⎢  3    4⎥
	// ⎢4.5    6⎥
	// ⎢  7    8⎥
	// ⎣  7   10⎦
	// median
	// ⎡ 1   2⎤
	// ⎢ 3   4⎥
	// ⎢ 3   6⎥
	// ⎢ 7   8⎥
	// ⎣ 7  10⎦
	// most_frequent
	// ⎡ 1   2⎤
	// ⎢ 3   4⎥
	// ⎢ 7   6⎥
	// ⎢ 7   8⎥
	// ⎣ 7  10⎦
	// imputation-of-missing-values:
	// ⎡                 4                   2⎤
	// ⎢                 6  3.6666666666666665⎥
	// ⎣                 7                   6⎦
}
