package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestSquareLoss(t *testing.T) {
	testLossDerivatives(t, "square")
}

func TestLogLoss(t *testing.T) {
	testLossDerivatives(t, "log")
}

func TestCrossEntropyLoss(t *testing.T) {
	testLossDerivatives(t, "cross-entropy")
}

func testLossDerivatives(t *testing.T, loss string) {
	nSamples, nOutputs := 100, 2
	unused(fd.Gradient, floats.Sum, fmt.Println)
	losser := NewLoss(loss)

	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
	YpredSlice := make([]float64, nSamples*nOutputs, nSamples*nOutputs)
	Ypred := mat.NewDense(nSamples, nOutputs, YpredSlice)
	gradSlice := make([]float64, nSamples*nOutputs, nSamples*nOutputs)
	gradSliceFd := make([]float64, nSamples*nOutputs, nSamples*nOutputs)
	Grad := mat.NewDense(nSamples, nOutputs, gradSlice)

	Ytrue.Apply(func(i int, j int, v float64) float64 {
		return rand.Float64()

	}, Ytrue)
	Ypred.Apply(func(i int, j int, v float64) float64 {

		return v + 1e-3*rand.NormFloat64()
	}, Ytrue)

	lossFd := func(YpredSlice []float64) float64 {
		return losser.Loss(Ytrue, mat.NewDense(nSamples, nOutputs, YpredSlice), nil)
	}
	//fmt.Printf("TestLoss %#v\n", activation)
	for i := 0; i < 10; i++ {
		// Alpha := 0.
		// L1Ratio := 0.
		losser.Loss(Ytrue, Ypred, Grad)
		fd.Gradient(gradSliceFd, lossFd, YpredSlice, &fd.Settings{Step: 1e-8})

		//fmt.Printf("%t %g %g\n", activation, grad.At(0, 0), dJ)
		if math.Abs(gradSlice[0]-gradSliceFd[0]) > 1e-2 {
			t.Errorf("%s %g fd:%g\n", loss, gradSlice[0], gradSliceFd[0])
		}
	}
}
