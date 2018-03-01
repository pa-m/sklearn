package linearModel

import (
	"fmt"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
)

/*


func TestCrossEntropy(t *testing.T){
  activation:=Sigmoid{}
  for i:=0;i<10;i++ {
    y:=1.
    x:=rand.NormFloat64()
    h:=activation.F(x)
    hprime:=activation.Fprime(x,h)
    fmt.Println(
      -y*hprime/h - (1-y) * (-hprime)/(1-h),
      (h-y)/(1.-h)* hprime/h,
    )
  }
}
*/

func TestLossDerivatives(t *testing.T) {
	nSamples := 100
	type pairType struct {
		Loss       Loss
		Activation Activation
	}
	for _, pair := range []pairType{
		//pairType{SquareLoss, Identity{}},
		{CrossEntropyLoss, Sigmoid{}},
	} {
		lossFunc := pair.Loss
		activation := pair.Activation
		//fmt.Printf("TestLoss %#v\n", activation)
		p := NewRandomLinearProblem(nSamples, 1, 1)
		p.Y.Apply(func(i int, j int, v float64) float64 {
			return activation.F(v)
		}, p.Y)
		unused(fd.Gradient, floats.Sum, fmt.Println)

		for i := 0; i < 10; i++ {
			Ytrue := p.Y
			X := p.X
			Theta := mat.NewDense(1, 1, []float64{rand.NormFloat64()})
			Ypred := mat.NewDense(nSamples, 1, nil)
			Ydiff := mat.NewDense(nSamples, 1, nil)
			Ytmp := mat.NewDense(nSamples, 1, nil)
			grad := mat.NewDense(1, 1, []float64{rand.NormFloat64()})
			Alpha := 0.
			L1Ratio := 0.

			J1 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, Ytmp, grad, Alpha, L1Ratio, nSamples, activation)
			eps := 1e-4
			Theta.Set(0, 0, Theta.At(0, 0)-2*eps)

			J0 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, Ytmp, grad, Alpha, L1Ratio, nSamples, activation)
			Theta.Set(0, 0, Theta.At(0, 0)+eps)
			lossFunc(Ytrue, X, Theta, Ypred, Ydiff, Ytmp, grad, Alpha, L1Ratio, nSamples, activation)

			dJ := (J1 - J0) / 2. / eps
			//fmt.Printf("%t %g %g\n", activation, grad.At(0, 0), dJ)
			if math.Abs(dJ-grad.At(0, 0)) > 1e-3 {
				t.Fail()
			}
		}

	}
}
