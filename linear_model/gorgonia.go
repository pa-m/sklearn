package linearModel

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	// use dot import for lisibility
	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Steper is an interface for gorgonia Step methods
type Steper interface {
	Step(model gg.Nodes) (err error)
}

func random(dt tensor.Dtype) interface{} {
	rand.Seed(13370)
	switch dt {
	case tensor.Float32:
		return rand.Float32()
	case tensor.Float64:
		return rand.Float64()
	default:
		panic("Unhandled dtype")
	}
}

func linearRegression(Float tensor.Dtype, xT, yT gg.Value) (m *gg.Node) {

	g := gg.NewGraph()
	Xshape := xT.(tensor.Tensor).Shape()
	x := gg.NewMatrix(g, Float, gg.WithShape(xT.Shape()...), gg.WithName("x"), gg.WithValue(xT))
	y := gg.NewVector(g, Float, gg.WithShape(yT.Shape()...), gg.WithName("y"), gg.WithValue(yT))
	m = gg.NewVector(g, Float, gg.WithShape(Xshape[1]), gg.WithName("m"), gg.WithInit(gg.Uniform(0., 1.)))
	Must := gg.Must
	pred := Must(gg.Mul(x, m))
	se := Must(gg.Square(Must(gg.Sub(pred, y))))
	cost := Must(gg.Mean(se))
	_, err := gg.Grad(cost, m)

	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	machine := gg.NewTapeMachine(g, gg.BindDualValues(m))

	defer runtime.GC()
	model := gg.Nodes{m}
	solver := gg.NewAdamSolver(gg.WithLearnRate(0.01), gg.WithClip(5)) // good idea to clip

	if gg.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	for i := 0; i < 5000; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
		}

		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}

		machine.Reset() // Reset is necessary in a loop like this
	}
	return
}

// Linear Regression Example
//
// The formula for a straight line is
//		y = mx + c
// We want to find an `m` and a `c` that fits the equation well. We'll do it in both float32 and float64 to showcase the extensibility of Gorgonia
/*func main() {
    // Float32
    linearRegression(Float32)

    // Float64
    linearRegression(Float64)

}*/
