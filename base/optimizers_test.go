package base

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

func assertPanics(t *testing.T, f func(), msg string) {
	defer func() {
		if r := recover(); r == nil {
			if msg == "" {
				msg = "should have panicked"
			}
			t.Error(msg)
		}
	}()
	f()
}
func TestNewOptimizer(t *testing.T) {
	assert.NotNil(t, NewOptimizer("sgd"))
	sgd := NewSGDOptimizer()
	assert.Equal(t, false, sgd.Adadelta || sgd.Adagrad || sgd.RMSProp || sgd.Adam)
	assert.Equal(t, true, NewAdagradOptimizer().Adagrad)
	assert.Equal(t, true, NewAdadeltaOptimizer().Adadelta)
	assert.Equal(t, true, NewRMSPropOptimizer().RMSProp)
	assert.Equal(t, true, NewAdamOptimizer().Adam)
	for _, opt := range []string{"adadelta", "adagrad", "adam", "rmsprop", "sgd"} {
		assert.Equal(t, 0, strings.Index(NewOptimizer(opt).String(), opt))
	}
	assert.Equal(t, true, sgd.Needs().Gradient)
	uses, err := sgd.Uses(optimize.Available{Grad: true})
	assert.Nil(t, err)
	assert.Equal(t, true, uses.Grad)
	sgd.SetTheta(mat.NewDense(1, 2, []float64{3, 4}))
	assert.Equal(t, 1, sgd.NFeatures)
	assert.Equal(t, 2, sgd.NOutputs)
	th := sgd.GetTheta()
	assert.Equal(t, []float64{3, 4}, th.RawMatrix().Data)

	sgd.UpdateParams(mat.NewDense(1, 2, []float64{0, 0}))
	assert.Equal(t, []float64{3, 4}, sgd.GetTheta().RawMatrix().Data)

	assertPanics(t, func() { NewOptimizer("xxx") }, "NewOptimizer xxx should panic") // should panic

	sgd = NewSGDOptimizer()
	uses, err = sgd.Uses(optimize.Available{})
	assert.NotNil(t, err)
	sgd.initLocal(&optimize.Location{X: []float64{0, 0}})
	assert.Equal(t, 2, sgd.NFeatures)
	assert.Equal(t, 1, sgd.NOutputs)
	assert.NotNil(t, sgd.Update)
}

// Commented out these tests because they are redundant with linear_model ones and theres an import cycle with linear_model

// import (
// 	"fmt"
// 	"golang.org/x/exp/rand"

// 	//"gonum.org/v1/gonum/mat"
// 	//"math"
// 	//"golang.org/x/exp/rand"
// 	"testing"

// 	//lm "github.com/pa-m/sklearn/linear_model"
// 	"gonum.org/v1/gonum/mat"
// 	"gonum.org/v1/gonum/optimize"
// 	//"time"
// )

// type OptimizerProblem struct {
// 	X, Theta, Y   *mat.Dense
// 	MiniBatchSize int
// }

// func NewRandomLinearProblem(nSamples, nFeatures, nOutputs int) *OptimizerProblem {
// 	X := mat.NewDense(nSamples, nFeatures, nil)
// 	X.Apply(func(i, j int, v float64) float64 {
// 		if j == 0 {
// 			return 1.
// 		}
// 		return rand.NormFloat64() * 10.
// 	}, X)
// 	TrueTheta := mat.NewDense(nFeatures, nOutputs, nil)
// 	TrueTheta.Apply(func(i, j int, v float64) float64 {
// 		return rand.NormFloat64() * 10.
// 	}, TrueTheta)
// 	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
// 	Ytrue.Product(X, TrueTheta)
// 	return &OptimizerProblem{X: X, Y: Ytrue, MiniBatchSize: nSamples}
// }

// func TestSGDSolver(t *testing.T) {
// 	fmt.Println("start TestSgdSolver")
// 	p := NewRandomLinearProblem(10000, 5, 4)
// 	p.MiniBatchSize = 100
// 	s := NewSGDOptimizer()
// 	s.StepSize = 1.
// 	s.Momentum = 0.98
// 	//s.GradientClipping = 10.
// 	testOptimizer(t, "SGDSolver", s, p)
// }

// func TestRMSProp(t *testing.T) {
// 	fmt.Println("start TestRMSProp")
// 	p := NewRandomLinearProblem(10000, 5, 4)
// 	p.MiniBatchSize = 50
// 	s := NewRMSPropOptimizer()
// 	s.StepSize = 1.
// 	//s.GradientClipping = 1.
// 	testOptimizer(t, "RMSProp", s, p)
// 	// for v := 0.; v <= 1.; v += .1 {
// 	// 	s.RMSPropGamma = v
// 	// 	ret := lm.LinFit(p.X, p.Y, &lm.LinFitOptions{MiniBatchSize: p.MiniBatchSize, Solver: s, Recorder: nil})
// 	// 	fmt.Println(v, ret.J, ret.RMSE)
// 	// }
// }

// func TestAdagrad(t *testing.T) {
// 	fmt.Println("start TestAdagrad")
// 	p := NewRandomLinearProblem(10000, 5, 4)
// 	p.MiniBatchSize = 50

// 	s := NewAdagradOptimizer()

// 	testOptimizer(t, "Adagrad", s, p)
// }

// func TestAdadelta(t *testing.T) {
// 	fmt.Println("start TestAdadelta")
// 	p := NewRandomLinearProblem(10000, 5, 4)
// 	p.MiniBatchSize = 100
// 	s := NewAdadeltaOptimizer()

// 	testOptimizer(t, "Adadelta", s, p)
// }
// func TestAdamSolver(t *testing.T) {
// 	p := NewRandomLinearProblem(10000, 5, 4)
// 	p.MiniBatchSize = 100
// 	s := NewAdamOptimizer()
// 	//s.GradientClipping = 10.
// 	// our problem is simplissime with a convex loss function so start with a big StepSize
// 	testOptimizer(t, "AdamSolver", s, p)
// }

// func testOptimizer(t *testing.T, name string, s Optimizer, p *OptimizerProblem) {
// 	X, Ytrue := p.X, p.Y
// 	//start := time.Now()
// 	recorder := optimize.NewPrinter()
// 	recorder = nil
// 	res := lm.LinFit(X, Ytrue, &lm.LinFitOptions{Epochs: 0, MiniBatchSize: p.MiniBatchSize, Tol: 1e-3, Solver: s, Alpha: 0., Recorder: recorder})
// 	if res.RMSE > 1e-2 {
// 		t.Errorf("%s NOK RMSE:%g", name, res.RMSE)
// 	} else {
// 		//fmt.Printf("%s ok. RMSE=%.9g epochs=%d elapsed=%s\n", name, res.RMSE, res.Epoch, time.Since(start))
// 	}
// }
