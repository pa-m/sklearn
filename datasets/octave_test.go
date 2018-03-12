package datasets

import (
	"math"
	"os"
	"testing"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestLoadOctaveBin(t *testing.T) {
	// v https://lists.gnu.org/archive/html/help-octave/2004-11/msg00068.html
	filename := os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/ex4data1.dat.gz"
	mats := LoadOctaveBin(filename)
	if _, ok := mats["y"]; !ok {
		t.Errorf("no y")
		return
	}
	X, Y := mats["X"], mats["y"]
	if math.Abs(mat.Sum(X)-262678.26016) > 1e-5 {
		t.Fail()
	}
	if mat.Sum(Y) != 27500 {
		t.Fail()
	}
	{
		e, a := 0., mat.Sum(X.Slice(0, 5000, 0, 1))
		if !floats.EqualWithinAbs(e, a, 1e-8) {
			t.Errorf("col 0 expected %g got %g", e, a)
		}

	}
	{
		e := -7.3744e-04
		a := X.At(0, 69)
		if !floats.EqualWithinAbs(e, a, 1e-8) {
			t.Errorf("0,69 expected %g got %g", e, a)
		}
	}
	{
		e := 4.8918e-05
		a := X.At(0, 353)
		if !floats.EqualWithinAbs(e, a, 1e-9) {
			t.Errorf("0,353 expected %g got %g", e, a)
		}
		// for i := 0; i < 5000; i++ {
		// 	for j := 0; j < 400; j++ {
		// 		a := X.At(i, j)
		// 		if floats.EqualWithinAbs(e, a, 1e-9) {
		// 			fmt.Printf("%d,%d %g\n", i, j, a)
		// 		}
		// 	}
		// }
	}
	{
		e := 62.227
		a := mat.Sum(X.Slice(0, 1, 0, 400))
		if !floats.EqualWithinAbs(e, a, 1e-3) {
			t.Errorf("line 0 expected %g got %g", e, a)
		}
	}
}
