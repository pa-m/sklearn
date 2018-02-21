package metrics

import "fmt"
import "testing"
import "math"

func TestR2Score(t *testing.T) {
	//1st example of sklearn metrics r2score
	yTrue := []float{3, -0.5, 2, 7}
	yPred := []float{2.5, 0.0, 2, 8}
	r2Score := R2Score(yTrue, yPred, nil, "varianceWeighted")
	fmt.Printf("r2Score=%g\n", r2Score)
	eps := 1e-3
	if math.Abs(0.948-r2Score) > eps {
		t.Fail()
	}
}

func TestMeanSquaredError(t *testing.T) {
	yTrue := []float{3, -0.5, 2, 7}
	yPred := []float{2.5, 0.0, 2, 8}
	eps := 1e-3
	if math.Abs(0.375-meanSquaredError(yTrue, yPred, nil)) > eps {
		t.Fail()
	}
}
