package metrics

import "fmt"
import "testing"
import "math"

func TestR2Score(t *testing.T) {
	//1st example of sklearn metrics r2score
	y_true := []float{3, -0.5, 2, 7}
	y_pred := []float{2.5, 0.0, 2, 8}
	r2_score := R2Score(y_true, y_pred, nil, "variance_weighted")
	fmt.Printf("r2_score=%g\n", r2_score)
	eps := 1e-3
	if math.Abs(0.948-r2_score) > eps {
		t.Fail()
	}
}
