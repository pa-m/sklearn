package neighbors

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestRectangle(t *testing.T) {
	R := NewRectangle([]float64{3, 4}, []float64{1, 2})
	expected := "<Rectangle 1 3, 2 4>"
	actual := R.String()
	if expected != actual {
		t.Errorf("expected %s, got %s", expected, actual)
	}
	if R.Volume() != 4. {
		t.Errorf("wrong volume")
	}
	if math.Abs(2.237-R.MinDistancePoint([]float64{0, 0}, 2)) > 1.e-3 {
		t.Error("err MinDistancePoint")
	}
	if math.Abs(5-R.MaxDistancePoint([]float64{0, 0}, 2)) > 1.e-3 {
		t.Error("err MaxDistancePoint")
	}
}

func ExampleKDTree() {
	// v https://github.com/scipy/scipy/blob/v1.1.0/scipy/spatial/kdtree.py
	X := mat.NewDense(30, 2, nil)
	for i := 0; i < 5; i++ {
		for j := 0; j < 6; j++ {
			X.Set(i*6+j, 0, float64(i))
			X.Set(i*6+j, 1, float64(j+2))
		}
	}
	//fmt.Println(mat.Formatted(X))
	tree := NewKDTree(X, 1)
	pts := mat.NewDense(2, 2, []float64{0, 0, 2.1, 2.9})
	distances, indices := tree.Query(pts, 1, 1e-15, 2, math.Inf(1))
	fmt.Printf("%.6f\n", mat.Formatted(distances.T()))
	fmt.Println(mat.Formatted(indices.T()))
	// Output:
	// [2.000000  0.141421]
	// [ 0  13]
}
