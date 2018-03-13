package neuralNetwork

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

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

	Z.Apply(func(_, _ int, _ float64) float64 { return rand.Float64() }, Z)

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
