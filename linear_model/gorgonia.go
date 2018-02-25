package linearModel

import (
	"fmt"
	"log"
	"runtime"
	// use dot import for lisibility
	"github.com/pa-m/sklearn/base"
	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var Float = gg.Float64

// LinearRegressionGorgonia uses gorgonia
type LinearRegressionGorgonia struct {
	LinearModel
	base.RegressorMixin
	LearningRate, Tol float
}

// NewLinearRegressionGorgonia
func NewLinearRegressionGorgonia() *LinearRegressionGorgonia {
	return &LinearRegressionGorgonia{LinearModel: LinearModel{FitIntercept: true}, LearningRate: .1, Tol: 1e-7}
}

// LinearRegressionGorgonia is a linear regression using gorgonia
func (regr *LinearRegressionGorgonia) Fit(X0 [][]float, y0 []float) *LinearRegressionGorgonia {
	var X, Y, XOffset, yOffset, XScale = preprocessData(
		X0, y0, regr.FitIntercept, regr.Normalize)
	regr.XOffset = XOffset
	regr.XScale = XScale
	Float := gg.Float64

	g := gg.NewGraph()
	Xshape := []int{len(X), len(X[0])}
	x := gg.NewMatrix(g, Float, gg.WithShape(Xshape...), gg.WithName("x"), gg.WithValue(ToTensor(X)))
	y := gg.NewVector(g, Float, gg.WithShape(Xshape[0]), gg.WithName("y"), gg.WithValue(ToTensor(Y)))
	w := gg.NewVector(g, Float, gg.WithShape(Xshape[1]), gg.WithName("w"), gg.WithInit(gg.Uniform(0., 1.)))
	Must := gg.Must
	pred := Must(gg.Mul(x, w))
	cost := Must(gg.Mean(Must(gg.Square(Must(gg.Sub(pred, y))))))
	_, err := gg.Grad(cost, w)

	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	machine := gg.NewTapeMachine(g, gg.BindDualValues(w))

	defer runtime.GC()
	model := gg.Nodes{w}
	solver := gg.NewAdamSolver(gg.WithLearnRate(regr.LearningRate), gg.WithClip(5)) // good idea to clip

	if gg.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	for i := 0; i < 5000; i++ {
		if err = machine.RunAll(); err != nil {
			break
		}
		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}
		//fmt.Println(i, cost.Value())
		machine.Reset() // Reset is necessary in a loop like this
		if cost.Value().Data().(float) < regr.Tol*regr.Tol {
			break
		}

	}

	regr.Coef = w.Value().Data().([]float)
	regr.SetIntercept(XOffset, yOffset, XScale)
	return regr

}

// Predict
func (regr *LinearRegressionGorgonia) Predict(X [][]float) (yMean []float) {
	yMean = regr.DecisionFunction(X)
	return
}

// --------
// LinearRegressionGorgonia uses gorgonia
type LinearRegressionGorgonia2 struct {
	LinearModel2
	base.RegressorMixin2
	nOutputs          int
	LearningRate, Tol float
}

// NewLinearRegressionGorgonia
func NewLinearRegressionGorgonia2() *LinearRegressionGorgonia2 {
	return &LinearRegressionGorgonia2{LinearModel2: LinearModel2{FitIntercept: true}, LearningRate: .1, Tol: 1e-7}
}

// LinearRegressionGorgonia is a linear regression using gorgonia
func (regr *LinearRegressionGorgonia2) Fit(X0, y0 [][]float) *LinearRegressionGorgonia2 {
	regr.nOutputs = len(y0[0])
	Float := gg.Float64

	g := gg.NewGraph()
	Xshape := []int{len(X0), len(X0[0])}
	Yshape := []int{len(y0), len(y0[0])}
	dMust := func(t *tensor.Dense, e error) *tensor.Dense {
		check(e)
		return t
	}
	xT := ToDense(X0)
	if regr.FitIntercept {
		ones := tensor.Ones(Float, Xshape[0], 1)
		xT = dMust(ones.Hstack(xT))
	}
	x := gg.NewMatrix(g, Float, gg.WithShape(xT.Shape()...), gg.WithName("x"), gg.WithValue(xT))
	Must := gg.Must
	var y, w, pred, cost *gg.Node
	if regr.nOutputs == 1 {
		backing := make([]float, len(y0), len(y0))
		for i, yi := range y0 {
			backing[i] = yi[0]
		}
		y = gg.NewVector(g, Float, gg.WithShape(xT.Shape()[0]), gg.WithName("y"), gg.WithValue(ToTensor(backing)))
		w = gg.NewVector(g, Float, gg.WithShape(xT.Shape()[1]), gg.WithName("w"), gg.WithInit(gg.Uniform(0., 1.)))
		fmt.Println("x", x.Shape(), "w", w.Shape(), "y", y.Shape())
		pred = Must(gg.Mul(x, w))
		cost = Must(gg.Mean(Must(gg.Square(Must(gg.Sub(pred, y)))), 0))

	} else {
		y = gg.NewMatrix(g, Float, gg.WithShape(Yshape...), gg.WithName("y"), gg.WithValue(ToTensor(y0)))
		w = gg.NewMatrix(g, Float, gg.WithShape(xT.Shape()[1], Yshape[1]), gg.WithName("w"), gg.WithInit(gg.Uniform(0., 1.)))
		pred = Must(gg.Mul(x, w))
		cost = Must(gg.Mean(Must(gg.Square(Must(gg.Sub(pred, y)))), 0, 1))

	}
	_, err := gg.Grad(cost, w)
	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	machine := gg.NewTapeMachine(g, gg.BindDualValues(w))

	defer runtime.GC()
	model := gg.Nodes{w}
	solver := gg.NewAdamSolver(gg.WithLearnRate(regr.LearningRate), gg.WithClip(5)) // good idea to clip

	if gg.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	for i := 0; i < 5000; i++ {
		if err = machine.RunAll(); err != nil {
			break
		}
		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}
		//fmt.Println(i, cost.Value())
		machine.Reset() // Reset is necessary in a loop like this
		//fmt.Println(i, cost.Value())
		if cost.Value().Data().(float) < regr.Tol*regr.Tol {
			break
		}

	}
	c := w.Value().Data().([]float)
	regr.Coef = make([][]float, Xshape[1], Xshape[1])
	regr.Intercept = make([]float, Yshape[1])
	ifeat0 := 0
	if regr.FitIntercept {
		ifeat0 = 1
		for iout := 0; iout < Yshape[1]; iout++ {
			regr.Intercept[iout] = c[iout]
		}
	}
	for ifeat := ifeat0; ifeat < xT.Shape()[1]; ifeat++ {

		regr.Coef[ifeat-ifeat0] = make([]float, Yshape[1], Yshape[1])
		for iout := 0; iout < Yshape[1]; iout++ {
			regr.Coef[ifeat-ifeat0][iout] = c[ifeat*Yshape[1]+iout]
		}
	}
	return regr

}

// Predict
func (regr *LinearRegressionGorgonia2) Predict(X [][]float) (yMean [][]float) {
	nSamples := len(X)
	XC, err := tensor.MatMul(ToDense(X), ToDense(regr.Coef))
	check(err)
	XCplusI, err := tensor.Add(ToDense([][]float{regr.Intercept}), XC)
	check(err)
	yMean = make([][]float, nSamples, nSamples)
	for i := range yMean {
		yMean[i] = make([]float, regr.nOutputs, regr.nOutputs)
		for j := range yMean[i] {
			v, err := XCplusI.At(i, j)
			check(err)
			yMean[i][j] = v.(float)
		}
	}
	return
}

// --------
func check(err error) {
	if err != nil {
		panic(err)
	}
}

func ToTensor(X interface{}) tensor.Tensor {
	switch v := X.(type) {
	case []float:
		return tensor.New(tensor.WithShape(len(v)), tensor.WithBacking(v))
	case [][]float:
		b := make([]float, len(v)*len(v[0]))
		k := 0
		for _, vi := range v {
			for _, vij := range vi {
				b[k] = vij
				k++
			}
		}
		return tensor.New(tensor.WithShape(len(v), len(v[0])), tensor.WithBacking(b))
	default:
		panic("[]float or [][]float expected")
	}
}

func ToDense(X interface{}) *tensor.Dense {
	switch v := X.(type) {
	case []float:
		return tensor.NewDense(Float, []int{len(v), 1}, tensor.WithBacking(v))
	case [][]float:
		b := make([]float, len(v)*len(v[0]))
		k := 0
		for _, vi := range v {
			for _, vij := range vi {
				b[k] = vij
				k++
			}
		}
		return tensor.New(tensor.WithShape(len(v), len(v[0])), tensor.WithBacking(b))
	default:
		panic("[]float or [][]float expected")
	}
}
