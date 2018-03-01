package linearModel

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Hinge loss , for SVMs, h=-1 +1. v https://en.wikipedia.org/wiki/Hinge_loss
// F: math.Max(0.,1.-h*y)
// Fprime: if 1. > h*y{return -y*hprime}else {return 0.}

// Loss puts cost in J and cost gradient in grad.
// Ytrue, X, Theta must be passed in
// Ypred,Ydiff,Ytmp are temporary matrices passed in here to avoid reallocations. nothing to initialize for them except storage
// Alpha and L1Ratio are for regularization
type Loss func(Ytrue, X mat.Matrix, Theta, Ypred, Ydiff, Ytmp, grad *mat.Dense, Alpha, L1Ratio float, nSamples int, activation Activation) (J float64)

// SquareLoss, Quadratic Loss, for regressions
// Ytrue, X, Theta must be passed in
// Ypred,Ydiff,Ytmp are temporary matrices passed in here to avoid reallocations. nothing to initialize for them except storage
// Alpha, L1Ratio are regularization parameters
// J: mat.Pow(h-y,2)/2
// grad:  hprime*X*(h-y)
func SquareLoss(Ytrue, X mat.Matrix, Theta, Ypred, Ydiff, Ytmp, grad *mat.Dense, Alpha, L1Ratio float, nSamples int, activation Activation) (J float64) {
	Ypred.Mul(X, Theta)
	Ypred.Apply(func(i, o int, xtheta float64) float64 { return activation.F(xtheta) }, Ypred)
	Ydiff.Sub(Ypred, Ytrue)
	Ytmp.MulElem(Ydiff, Ydiff)
	// compute hprime
	Ytmp.Apply(func(i, o int, ypred float64) float64 {
		return activation.Fprime(ypred)
	}, Ypred)
	Ytmp.MulElem(Ytmp, Ydiff)
	// put into grad
	J = mat.Sum(Ytmp)
	grad.Mul(X.T(), Ytmp)
	// add regularization to cost and grad
	if Alpha > 0. {
		L1, L2 := 0., 0.
		grad.Apply(func(j, o int, g float64) float64 {
			c := Theta.At(j, o)
			L1 += math.Abs(c)
			L2 += c * c
			g += Alpha * (L1Ratio*sgn(c) + (1.-L1Ratio)*c)
			return g
		}, grad)
		J += Alpha * (L1Ratio*L1 + (1. - L1Ratio*L2))
	}

	J /= 2. * float64(nSamples)
	grad.Scale(1./float64(nSamples), grad)
	return
}

// Cross entropy Loss
// J: -y*math.Log(h)-(1.-y)*log(1.-h)
// grad:  hprime*(-y/h + (1-y)/(1-h))

func CrossEntropyLoss(Ytrue, X mat.Matrix, Theta, Ypred, Ydiff, Ytmp, grad *mat.Dense, Alpha, L1Ratio float, nSamples int, activation Activation) (J float64) {
	Ypred.Mul(X, Theta)
	Ypred.Apply(func(i, o int, xtheta float64) float64 { return activation.F(xtheta) }, Ypred)
	Ydiff.Sub(Ypred, Ytrue)
	J = 0.
	Ytmp.Apply(func(i int, o int, h float64) float64 {
		eps := 1e-10
		y := Ytrue.At(i, o)
		if h == 0. {
			h += eps
		} else if h == 1. {
			h -= eps
		}
		J += -y*math.Log(h) - (1.-y)*math.Log(1.-h)
		if math.IsNaN(J) {
			panic(fmt.Errorf("J Nan after -y*math.Log(h) - (1.-y)*math.Log(1.-h). y=%g h=%g", y, h))
		}
		if math.IsInf(J, 1) {
			panic(fmt.Errorf("CrossEntropyLoss J Inf h=%g y=%g", h, y))
		}
		return h
	}, Ypred)
	grad.Mul(X.T(), Ydiff)
	// add regularization to cost and grad
	if Alpha > 0. {
		L1, L2 := 0., 0.
		grad.Apply(func(j, o int, g float64) float64 {
			c := Theta.At(j, o)
			L1 += math.Abs(c)
			L2 += c * c
			g += Alpha * (L1Ratio*sgn(c) + (1.-L1Ratio)*c)
			return g
		}, grad)
		J += Alpha * (L1Ratio*L1 + (1. - L1Ratio*L2))
	}
	J /= float64(nSamples)
	grad.Scale(1./float64(nSamples), grad)
	if math.IsNaN(J) {
		panic("J Nan")
	}
	return
}

// J=-y*log(h)-(1-y)*log(1-h)
// J = (sym)
//
//          ⎛    1    ⎞               ⎛        1    ⎞
//   - y⋅log⎜─────────⎟ - (-y + 1)⋅log⎜1 - ─────────⎟
//          ⎜     -X⋅θ⎟               ⎜         -X⋅θ⎟
//          ⎝1 + ℯ    ⎠               ⎝    1 + ℯ    ⎠
//
// >> diff(J,theta)
// ans = (sym)
//
//          -X⋅θ                     -X⋅θ
//     X⋅y⋅ℯ             X⋅(-y + 1)⋅ℯ
//   - ───────── + ────────────────────────────
//          -X⋅θ                              2
//     1 + ℯ       ⎛        1    ⎞ ⎛     -X⋅θ⎞
//                 ⎜1 - ─────────⎟⋅⎝1 + ℯ    ⎠
//                 ⎜         -X⋅θ⎟
//                 ⎝    1 + ℯ    ⎠
//

func sgn(c float64) float64 {
	if c < 0. {
		return -1.
	}
	if c > 0. {
		return 1.
	}
	return 0.
}
