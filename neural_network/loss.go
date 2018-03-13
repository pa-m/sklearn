package neuralNetwork

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type lossBaseStruct struct{}

// LossFunctions is the interface for matLoss (matSquareLoss,...)
type LossFunctions interface {
	Loss(Ytrue, Ypred, Grad *mat.Dense) float64
}

type squareLoss struct{ lossBaseStruct }

func (squareLoss) Loss(Ytrue, Ypred, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	// J:=(h-y)^2/2
	// Ydiff := matSub{A: Ypred, B: Ytrue}
	// J := metrics.MeanSquaredError(Ytrue, Ypred, nil, "").At(0, 0)
	J := matx{}.SumApplied2(Ytrue, Ypred, func(y, h float64) float64 { yd := h - y; return yd * yd / 2. })
	// Grad:=(h-y)
	if Grad != nil {
		//Grad.Scale(1./float64(nSamples), Ydiff)
		matx{Dense: Grad}.CopyScaledApplied2(Ytrue, Ypred, 1./float64(nSamples), func(y, h float64) float64 { return h - y })
	}
	return J
}

type logLoss struct{ lossBaseStruct }

func (logLoss) Loss(Ytrue, Ypred, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	// J:=-y log(h)
	//J := -mat.Sum(matMulElem{A: Ytrue, B: base.MatApply1{Matrix: Ypred, Func: math.Log}}) / float64(nSamples)
	J := matx{}.SumApplied2(Ytrue, Ypred, func(y, h float64) float64 { return -y * math.Log(h) }) / float64(nSamples)
	// Grad:=-y/h
	if Grad != nil {
		Gfun := func(y, h float64) float64 { return -y / h }
		//Grad.Scale(1./float64(nSamples), matApply2{A: Ytrue, B: Ypred, Func: Gfun})
		matx{Dense: Grad}.CopyScaledApplied2(Ytrue, Ypred, 1./float64(nSamples), Gfun)
	}
	return J
}

type crossEntropyLoss struct{ lossBaseStruct }

func (crossEntropyLoss) Loss(Ytrue, Ypred, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	// J:=-y log(h)-(1-y) log(1-h)
	Jfun := func(y, h float64) float64 {
		eps := 1e-15
		if h <= 0 {
			h = eps
		} else if h >= 1. {
			h = 1 - eps
		}
		return -y*math.Log(h) - (1.-y)*math.Log(1.-h)
	}
	//fmt.Printf("h11=%f J11=%f\n", Ypred.At(0, 0), Jfun(Ytrue.At(0, 0), Ypred.At(0, 0))/float64(nSamples))
	J := matx{}.SumApplied2(Ytrue, Ypred, Jfun) / float64(nSamples)
	if Grad != nil {
		// Grad:=-y/h+(1-y)/(1-h)
		Gfun := func(y, h float64) float64 {
			eps := 1e-12
			if h <= 0 {
				h = eps
			} else if h >= 1. {
				h = 1 - eps
			}
			return -y/h + (1-y)/(1-h)
		}
		//Grad.Scale(1./float64(nSamples), matApply2{A: Ytrue, B: Ypred, Func: Gfun})
		matx{Dense: Grad}.CopyScaledApplied2(Ytrue, Ypred, 1./float64(nSamples), Gfun)
	}
	return J
}

// SupportedLoss are the map[string]Losser of available matrix loss function providers
var SupportedLoss = map[string]LossFunctions{
	"square":        squareLoss{},
	"log":           logLoss{},
	"cross-entropy": crossEntropyLoss{},
}

// NewLoss creates a LossFunctions by its name
func NewLoss(name string) LossFunctions {
	loss, ok := SupportedLoss[name]
	if !ok {
		panic(fmt.Errorf("loss %s is unknown", name))
	}
	return loss
}
