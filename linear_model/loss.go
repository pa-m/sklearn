package linearModel

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Hinge loss , for SVMs, h=-1 +1. v https://en.wikipedia.org/wiki/Hinge_loss
// F: math.Max(0.,1.-h*y)
// Fprime: if 1. > h*y{return -y*hprime}else {return 0.}

// Loss puts cost in J and cost gradient in grad.
// Ytrue, X, Theta must be passed in
// Ypred,Ydiff,Ytmp are temporary matrices passed in here to avoid reallocations. nothing to initialize for them except storage
// Alpha and L1Ratio are for regularization
// Loss derivative is dJWrtTheta=dJWrth*dhWrtz*X
type Loss func(Ytrue, X, Theta mat.Matrix, Ypred, Ydiff, grad *mat.Dense, Alpha, L1Ratio float64, nSamples int, activation Activation) (J float64)

// LossFunctions is the map of implemented loss functions
var LossFunctions = map[string]Loss{"square": SquareLoss, "log": LogLoss, "cross-entropy": CrossEntropyLoss}

// SquareLoss Quadratic Loss, for regressions
// Ytrue, X, Theta must be passed in
// Ypred,Ydiff,Ytmp are temporary matrices passed in here to avoid reallocations. nothing to initialize for them except storage
// Alpha, L1Ratio are regularization parameters
// J: mat.Pow(h-y,2)/2
// grad:  hprime*(h-y)
//
func SquareLoss(Ytrue, X, Theta mat.Matrix, Ypred, Ydiff, grad *mat.Dense, Alpha, L1Ratio float64, nSamples int, activation Activation) (J float64) {
	Ypred.Mul(X, Theta)
	Ypred.Apply(func(i, o int, xtheta float64) float64 { return activation.F(xtheta) }, Ypred)
	Ydiff.Sub(Ypred, Ytrue)
	J = 0.
	Ydiff.Apply(func(_ int, _ int, ydiff float64) float64 {
		J += ydiff * ydiff
		return ydiff
	}, Ydiff)
	// put into grad
	if _, ok := activation.(Identity); ok {
		grad.Mul(X.T(), Ydiff) //<- for identity only

	} else {
		grad.Apply(func(j, o int, theta float64) float64 {
			g := 0.
			for i := 0; i < nSamples; i++ {
				h := Ypred.At(i, o)
				g += Ydiff.At(i, o) * X.At(i, j) * activation.Fprime(h)
			}
			return g
		}, Theta)
	}
	// add regularization to cost and grad
	if Alpha > 0. {
		L1, L2 := 0., 0.
		grad.Apply(func(j, o int, g float64) float64 {
			// here we count feature 0 (not ones) in regularization
			if j >= 0 {
				c := Theta.At(j, o)

				L1 += math.Abs(c)
				L2 += c * c
				g += Alpha * (L1Ratio*sgn(c) + (1.-L1Ratio)*c)
			}
			return g
		}, grad)
		J += Alpha * (L1Ratio*L1 + (1.-L1Ratio)*L2)
	}

	J /= 2. * float64(nSamples)
	grad.Scale(1./float64(nSamples), grad)
	return
}

// LogLoss for one versus rest classifiers
func LogLoss(Ytrue, X, Theta mat.Matrix, Ypred, Ydiff, grad *mat.Dense, Alpha, L1Ratio float64, nSamples int, activation Activation) (J float64) {
	Ypred.Mul(X, Theta)
	Ypred.Apply(func(i, o int, xtheta float64) float64 { return activation.F(xtheta) }, Ypred)
	Ydiff.Sub(Ypred, Ytrue)
	J = 0.
	Ypred.Apply(func(i int, o int, hpred float64) float64 {
		eps := 1e-10
		y := Ytrue.At(i, o)
		h := hpred
		if hpred == 0. {
			h += eps
		} else if h == 1. {
			h -= eps
		}
		J += -y * math.Log(h)
		if math.IsNaN(J) {
			panic(fmt.Errorf("J Nan after -y*math.Log(h) . y=%g h=%g", y, h))
		}
		if math.IsInf(J, 1) {
			panic(fmt.Errorf("LogLoss J Inf h=%g y=%g", h, y))
		}
		return hpred
	}, Ypred)
	//grad.Mul(X.T(), Ydiff)
	grad.Apply(func(j, o int, theta float64) float64 {
		g := 0.
		for i := 0; i < nSamples; i++ {
			h := Ypred.At(i, o)
			g += -Ytrue.At(i, o) * activation.Fprime(h) / h
		}
		return g
	}, Theta)

	// add regularization to cost and grad
	if Alpha > 0. {
		L1, L2 := 0., 0.
		grad.Apply(func(j, o int, g float64) float64 {
			// we dont count feature 0 (ones) in regularization
			if j > 0 {
				c := Theta.At(j, o)
				L1 += math.Abs(c)
				L2 += c * c / 2.
				g += Alpha * (L1Ratio*sgn(c) + (1.-L1Ratio)*c)
			}
			return g
		}, grad)
		J += Alpha * (L1Ratio*L1 + (1.-L1Ratio)*L2)
	}
	J /= float64(nSamples)
	grad.Scale(1./float64(nSamples), grad)
	if math.IsNaN(J) {
		panic("J Nan")
	}
	return
}

// CrossEntropyLoss is the loss for LogisticRegression and Classifiers
// J: -y*math.Log(h)-(1.-y)*log(1.-h)
// grad:  hprime*(-y/h + (1-y)/(1-h))
//
func CrossEntropyLoss(Ytrue, X, Theta mat.Matrix, Ypred, Ydiff, grad *mat.Dense, Alpha, L1Ratio float64, nSamples int, activation Activation) (J float64) {
	Ypred.Mul(X, Theta)
	Ypred.Apply(func(i, o int, xtheta float64) float64 { return panicIfNaN(activation.F(xtheta)) }, Ypred)
	Ydiff.Sub(Ypred, Ytrue)
	J = 0.
	// compute J = Sum -y*math.Log(h) - (1.-y)*math.Log(1.-h)
	Ypred.Apply(func(i int, o int, hpred float64) float64 {
		eps := 1e-10
		h := hpred
		if h <= 0. {
			h = eps
		} else if h >= 1. {
			h = 1. - eps
		}
		y := Ytrue.At(i, o)
		J += -y*math.Log(h) - (1.-y)*math.Log(1.-h)
		if math.IsNaN(J) {
			panic(fmt.Errorf("J Nan after -y*math.Log(h) - (1.-y)*math.Log(1.-h). y=%g h=%g", y, h))
		}
		if math.IsInf(J, 1) {
			panic(fmt.Errorf("CrossEntropyLoss J Inf h=%g y=%g", h, y))
		}
		return hpred
	}, Ypred)
	if _, ok := activation.(Logistic); ok {
		grad.Mul(X.T(), Ydiff)
	} else {
		// // for Logistic activation only
		grad.Apply(func(j, o int, theta float64) float64 {
			g := 0.
			for i := 0; i < nSamples; i++ {
				h := Ypred.At(i, o)
				y := Ytrue.At(i, o)
				hprime := activation.Fprime(h)
				if y == 1. {
					g += -y * hprime / h
				} else if y == 0. {
					g += (1. - y) * hprime / (1. - h)
				} else {
					g += -y*hprime/h + (1.-y)*hprime/(1.-h)
				}
				if math.IsNaN(g) {
					panic(fmt.Errorf("g is NaN h=%g y=%g ", h, y))
				}
			}
			return g
		}, Theta)
	}
	// add regularization to cost and grad
	if Alpha > 0. {
		L1, L2 := 0., 0.
		grad.Apply(func(j, o int, g float64) float64 {
			// we dont count feature 0 (ones) in regularization
			if j > 0 || o > 0 {
				c := Theta.At(j, o)
				L1 += math.Abs(c)
				L2 += c * c / 2.
				g += Alpha * (L1Ratio*sgn(c) + (1.-L1Ratio)*c)
			}
			return g
		}, grad)
		//fmt.Printf("J=1/%d * { %g + %g * ( %g*%g +%g*%g) }\n", nSamples, J, Alpha, L1Ratio, L1, 1.-L1Ratio, L2)
		J += Alpha * (L1Ratio*L1 + (1.-L1Ratio)*L2)
		//fmt.Printf("=%g\n", J)
	}
	J /= float64(nSamples)
	//fmt.Printf("/%d =>%g\n", nSamples, J)
	grad.Scale(1./float64(nSamples), grad)
	return
}

// J=-y*log(h)-(1-y)*log(1-h)
// J =
//
//          ⎛    1    ⎞               ⎛        1    ⎞
//   - y⋅log⎜─────────⎟ - (-y + 1)⋅log⎜1 - ─────────⎟
//          ⎜     -X⋅θ⎟               ⎜         -X⋅θ⎟
//          ⎝1 + ℯ    ⎠               ⎝    1 + ℯ    ⎠
//
// >> diff(J,theta)
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

// TODO Hinge,HingeSmoothed,Huber, v https://en.wikipedia.org/wiki/Hinge_loss

func sgn(c float64) float64 {
	if c < 0. {
		return -1.
	}
	if c > 0. {
		return 1.
	}
	return 0.
}

func panicIfNaN(v float64) float64 {
	if math.IsNaN(v) {
		panic("NaN")
	}
	return v
}
