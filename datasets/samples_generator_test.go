package datasets

import (
	"fmt"
)

func ExampleMakeRegression() {
	X, Y, _ := MakeRegression(map[string]interface{}{"n_samples": 200, "n_features": 3, "n_informative": 2, "n_targets": 2,
		"bias":    []float64{1., 2.},
		"shuffle": true,
	})
	xr, xc := X.Dims()
	fmt.Println("X", xr, xc)
	yr, yc := Y.Dims()
	fmt.Println("Y", yr, yc)
	// Output:
	// X 200 3
	// Y 200 2

}

func ExampleMakeBlobs() {
	X, Y := MakeBlobs(&MakeBlobsConfig{})
	rx, cx := X.Dims()
	ry, cy := Y.Dims()
	fmt.Printf("rx=%d cx=%d ry=%d cy=%d\n", rx, cx, ry, cy)
	// Output:
	// rx=100 cx=2 ry=100 cy=1
}
