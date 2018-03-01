package linearModel

import (
	"fmt"
	"log"
	"math"
	"runtime"
	// use dot import for lisibility
	//"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Float is gorgonia's Float64
var Float = gg.Float64

// LinearRegressionGorgonia is a multioutput libear regression using gorgonia
type LinearRegressionGorgonia struct {
	LinearModel
	nOutputs, Epochs                  int
	LearningRate, Tol, Alpha, L1Ratio float
}

// NewLinearRegressionGorgonia create a *LinearRegressionGorgonia with good defaults
func NewLinearRegressionGorgonia() *LinearRegressionGorgonia {
	return &LinearRegressionGorgonia{LinearModel: LinearModel{FitIntercept: true}, LearningRate: .1, Tol: 1e-7}
}

// Fit lears coef and intercept for a *LinearRegressionGorgonia
func (regr *LinearRegressionGorgonia) Fit(X0, y0 *mat.Dense) Regressor {
	Float := gg.Float64

	g := gg.NewGraph()
	r, c := X0.Dims()
	Xshape := []int{r, c}
	nSamples := r
	r, c = y0.Dims()
	Yshape := []int{r, c}
	regr.nOutputs = c
	dMust := func(t *tensor.Dense, e error) *tensor.Dense {
		check(e)
		return t
	}
	xT := tensor.FromMat64(X0)
	if regr.FitIntercept {
		ones := tensor.Ones(Float, Xshape[0], 1)
		xT = dMust(ones.Hstack(xT))
	}
	x := gg.NewMatrix(g, Float, gg.WithShape(xT.Shape()...), gg.WithName("x"), gg.WithValue(xT))
	Must := gg.Must
	var y, w, pred, cost *gg.Node
	if regr.nOutputs == 1 {
		y = gg.NewVector(g, Float, gg.WithShape(xT.Shape()[0]), gg.WithName("y"), gg.WithValue(tensor.FromMat64(y0)))
		w = gg.NewVector(g, Float, gg.WithShape(xT.Shape()[1]), gg.WithName("w"), gg.WithInit(gg.Uniform(0., 1.)))
		fmt.Println("x", x.Shape(), "w", w.Shape(), "y", y.Shape())
		pred = Must(gg.Mul(x, w))
		cost = Must(gg.Mean(Must(gg.Square(Must(gg.Sub(pred, y)))), 0))

	} else {
		y = gg.NewMatrix(g, Float, gg.WithShape(Yshape...), gg.WithName("y"), gg.WithValue(ToDenseTensor(y0)))
		w = gg.NewMatrix(g, Float, gg.WithShape(xT.Shape()[1], Yshape[1]), gg.WithName("w"), gg.WithInit(gg.Uniform(0., 1.)))
		pred = Must(gg.Mul(x, w))
		cost = Must(gg.Mean(Must(gg.Square(Must(gg.Sub(pred, y)))), 0, 1))

	}
	if regr.Alpha >= 0. {
		L1 := Must(gg.Mul(gg.NewConstant(regr.Alpha*regr.L1Ratio/float64(2*nSamples)), Must(gg.Sum(Must(gg.Abs(w))))))
		cost = Must(gg.Add(cost, L1))
		L2 := Must(gg.Mul(gg.NewConstant(regr.Alpha*(1.-regr.L1Ratio)/float64(2*nSamples)), Must(gg.Sum(Must(gg.Square(w))))))
		cost = Must(gg.Add(cost, L2))
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
	if regr.Epochs <= 0 {
		regr.Epochs = 1e6 / nSamples
	}
	for i := 0; i < regr.Epochs; i++ {
		if err = machine.RunAll(); err != nil {
			break
		}
		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}
		//fmt.Println(i, cost.Value())
		machine.Reset() // Reset is necessary in a loop like this
		//fmt.Println(i, cost.Value())
		if math.Sqrt(cost.Value().Data().(float)) < regr.Tol {
			break
		}

	}
	wmat := mat.NewDense(xT.Shape()[1], Yshape[1], w.Value().Data().([]float))
	regr.Coef = mat.NewDense(Xshape[1], Yshape[1], nil)
	regr.Intercept = mat.NewDense(1, Yshape[1], nil)
	ifeat0 := 0
	if regr.FitIntercept {
		ifeat0 = 1
		regr.Intercept.Clone(wmat.RowView(0).T())
	}
	regr.Coef.Apply(func(j, o int, c float64) float64 { return wmat.At(j+ifeat0, o) }, regr.Coef)
	return regr

}

// Predict return predicted Ys for a list or Xs
func (regr *LinearRegressionGorgonia) Predict(X, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)
}

// --------
func check(err error) {
	if err != nil {
		panic(err)
	}
}

// ToDenseTensor converts to a *tensor.Dense
// accepts []float64 [][]float64 *mat.Dense mat.Matrix
func ToDenseTensor(X interface{}) *tensor.Dense {
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
	case *mat.Dense:
		return tensor.FromMat64(v)
	case mat.Matrix:
		return tensor.FromMat64(mat.DenseCopyOf(v))
	default:
		panic("[]float or [][]float expected")
	}
}
