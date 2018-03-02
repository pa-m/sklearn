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

func testLossDerivatives(t *testing.T, lossFunc Loss, activation Activation) {
	nSamples := 100
	unused(fd.Gradient, floats.Sum, fmt.Println)

	Ypred := mat.NewDense(nSamples, 1, nil)
	Ydiff := mat.NewDense(nSamples, 1, nil)
	grad := mat.NewDense(1, 1, nil)
	Theta := mat.NewDense(1, 1, nil)
	{
		p := NewRandomLinearProblem(nSamples, 1, 1)
		p.Y.Apply(func(i int, j int, v float64) float64 {
			return activation.F(v)
		}, p.Y)
		X := p.X
		Ytrue := p.Y
		//fmt.Printf("TestLoss %#v\n", activation)
		for i := 0; i < 10; i++ {
			Alpha := 0.
			L1Ratio := 0.

			theta0 := rand.NormFloat64()
			eps := 1e-2
			Theta.Set(0, 0, theta0+eps)
			J1 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation)
			Theta.Set(0, 0, theta0-eps)

			J0 := lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation)
			Theta.Set(0, 0, theta0)
			lossFunc(Ytrue, X, Theta, Ypred, Ydiff, grad, Alpha, L1Ratio, nSamples, activation)

			dJ := (J1 - J0) / 2. / eps
			//fmt.Printf("%t %g %g\n", activation, grad.At(0, 0), dJ)
			if math.Abs(dJ-grad.At(0, 0)) > 1e-3 {
				fmt.Printf("%t %g %g\n", activation, grad.At(0, 0), dJ)
				t.Fail()
			}
		}

	}
}

func TestSquareLoss(t *testing.T) {
	testLossDerivatives(t, SquareLoss, Identity{})
}

func TestCrossEntropyLoss(t *testing.T) {
	testLossDerivatives(t, CrossEntropyLoss, Sigmoid{})
}
