package neuralNetwork

import (
	"fmt"
	"math"
	"math/rand"

	"testing"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/diff/fd"
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

func TestIdentity(t *testing.T) {
	testActivationDerivatives(t, "identity")
}
func TestLogistic(t *testing.T) {
	testActivationDerivatives(t, "logistic")
}
func TestTanh(t *testing.T) {
	testActivationDerivatives(t, "tanh")
}
func TestReLU(t *testing.T) {
	testActivationDerivatives(t, "relu")
}

func testActivationDerivatives(t *testing.T, activation string) {
	nSamples, nOutputs := 1, 1
	if _, ok := SupportedActivations[activation]; !ok {
		t.Errorf("activation %s not in matActivations", activation)
		return
	}
	a, _ := SupportedActivations[activation]
	zSlice := make([]float64, nSamples*nOutputs)
	hSlice := make([]float64, nSamples*nOutputs)
	gradSlice := make([]float64, nSamples*nOutputs)
	gradSliceFd := make([]float64, nSamples*nOutputs)

	Z := mat.NewDense(nSamples, nOutputs, zSlice)
	H := mat.NewDense(nSamples, nOutputs, hSlice)
	G := mat.NewDense(nSamples, nOutputs, gradSlice)
	//Gfd := mat.NewDense(nSamples, nOutputs, gradSliceFd)

	Z.Copy(matApply0{Rows: nSamples, Columns: nOutputs, Func: rand.Float64})

	a.Func(Z, H)
	a.Grad(Z, H, G)

	for sample := 0; sample <= nSamples; sample++ {
		for output := 0; output < nOutputs; output++ {
			sample := rand.Intn(nSamples)
			fdFun := func(zSliceFd []float64) float64 {
				a.Func(mat.NewDense(nSamples, nOutputs, zSliceFd), H)
				h := H.At(sample, output)
				return h
			}
			fd.Gradient(gradSliceFd, fdFun, zSlice, &fd.Settings{Step: 1e-6})
			if math.Abs(G.At(sample, output)-gradSliceFd[0]) > 1e-2 {
				t.Errorf("%s z:%g g:%g fd:%g", activation, Z.At(sample, output), G.At(sample, output), gradSliceFd[0])
			}
		}
	}

}
