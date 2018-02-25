package linearModel

import (
	"fmt"
	"math"
	"math/rand"

	"testing"
	"time"
)

const (
	vecSize = 10000
)

// manually generate a fake dataset which is y=2x+random
func xy() (xBack [][]float, yBack []float) {
	/*var xBack, yBack interface{}

	xBack = tensor.Range(Float, 1, vecSize+1).([]float64)
	yBackC := tensor.Range(Float, 1, vecSize+1).([]float64)

	for i, v := range yBackC {
		yBackC[i] = v*2 + rand.Float64()
	}
	yBack = yBackC
	*/
	nFeatures := 3
	nOutputs := 1 // Grad can't handle fails with more than one output
	xBack = make([][]float, vecSize)
	yBack = make([]float, vecSize*nOutputs)
	for i := range yBack {
		xBack[i] = make([]float, nFeatures, nFeatures)

		xBack[i][0] = rand.Float64()*20. - 10.
		xBack[i][1] = rand.Float64()*20. - 10.
		xBack[i][2] = rand.Float64()*20. - 10.

		yBack[nOutputs*i] = 8 + 2*xBack[i][0] + 3*xBack[i][1] + xBack[i][2]
	}
	return
}

func TestLinearRegressionGorgonia(t *testing.T) {
	start := time.Now()
	X, Y := xy()
	for _, norm := range []bool{false, true} {
		m := NewLinearRegressionGorgonia()
		m.Normalize = norm
		m.Tol = 1e-6
		m.Fit(X, Y)
		elapsed := time.Since(start)
		fmt.Println("TestLinearRegressionGorgonia", elapsed)
		//fmt.Printf("TestLinearRegressionG(normalize=%v) %v: Intercept=%g Coef=%#v\nelapsed:%s\n", norm, Float, m.Intercept, m.Coef, elapsed)
		erry := 14. - m.Predict([][]float{[]float{1, 1, 1}})[0]
		if math.Abs(erry) > m.Tol {
			t.Errorf("TestLinearRegressionG Error:%g", erry)
		}
	}
}

// LinearRegressionGorgonia2 is a linear regr with sevelal outputs
func TestLinearRegressionGorgonia2(t *testing.T) {
	start := time.Now()
	X, Y1 := xy()
	Y := make([][]float, len(Y1))
	for i, yi := range Y1 {
		Y[i] = []float{yi, 2 * yi}
	}
	for _, norm := range []bool{false, true} {
		m := NewLinearRegressionGorgonia2()
		m.Normalize = norm
		m.Tol = 1e-6
		m.Fit(X, Y)
		elapsed := time.Since(start)
		//fmt.Printf("TestLinearRegressionG(normalize=%v) %v: Intercept=%v Coef=%#v\nelapsed:%s\n", norm, Float, m.Intercept, m.Coef, elapsed)
		fmt.Println("TestLinearRegressionGorgonia2", elapsed)
		ypred := m.Predict([][]float{[]float{1, 1, 1}})
		erry := 14. - ypred[0][0]
		if math.Abs(erry) > 2*m.Tol {
			t.Errorf("TestLinearRegressionG Error:%g", erry)
		}
		erry = 28. - ypred[0][1]
		if math.Abs(erry) > 2*m.Tol {
			t.Errorf("TestLinearRegressionG Error:%g", erry)
		}
	}
}
