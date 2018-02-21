package linearModel

import (
	"fmt"
	"math/rand"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"testing"
	"time"
)

const (
	vecSize = 10000
)

var Float = Float64

// manually generate a fake dataset which is y=2x+random
func xy() (x tensor.Tensor, y tensor.Tensor) {
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
	xBack := make([]float, vecSize*nFeatures)
	yBack := make([]float, vecSize*nOutputs)
	for i := range yBack {
		xBack[nFeatures*i] = rand.Float64()*20. - 10.
		xBack[nFeatures*i+1] = rand.Float64()*20. - 10.
		xBack[nFeatures*i+2] = rand.Float64()*20. - 10.

		yBack[nOutputs*i] = 2*xBack[nFeatures*i] + 3*xBack[nFeatures*i+1] + xBack[nFeatures*i+2]
	}

	//x = tensor.New(tensor.WithBacking(xBack), tensor.WithShape(vecSize, 2))
	x = tensor.NewDense(Float, []int{vecSize, nFeatures}, tensor.WithBacking(xBack))

	y = tensor.New(tensor.WithBacking(yBack), tensor.WithShape(vecSize))
	return
}

func TestLinearRegressionG(t *testing.T) {
	var xT, yT Value
	xT, yT = xy()
	start := time.Now()

	m := linearRegression(Float, xT, yT)
	elapsed := time.Since(start)
	fmt.Printf("TestLinearRegressionG %v: m=%#v\nelapsed:%s\n", Float, m.Value(), elapsed)

}
