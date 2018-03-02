package base

import (
	"fmt"
	//"gonum.org/v1/gonum/mat"
	//"math"
	//"math/rand"
	"testing"
	//"time"
)

func TestSGDSolver(t *testing.T) {
	fmt.Println("start TestSgdSolver")
	p := NewRandomLinearProblem(10000, 5, 5)
	s := NewSGDOptimizer()
	s.Adagrad = false
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = 0.01

	testSolver(t, "SGDSolver", s, p)
}

func TestRMSProp(t *testing.T) {
	fmt.Println("start TestRMSProp")
	p := NewRandomLinearProblem(10000, 5, 5)
	p.Normalize()
	p.MiniBatchSize = 1000
	s := NewRMSPropOptimizer()

	//s.RMSPropGamma = .9
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = 0.001

	testSolver(t, "RMSProp", s, p)
}

func TestAdagrad(t *testing.T) {
	fmt.Println("start TestAdagrad")
	p := NewRandomLinearProblem(10000, 5, 5)
	p.Normalize()
	p.MiniBatchSize = 100
	s := NewAdagradOptimizer()
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = 0.01

	testSolver(t, "Adagrad", s, p)
}

func TestAdadelta(t *testing.T) {
	fmt.Println("start TestAdadelta")
	p := NewRandomLinearProblem(10000, 5, 5)
	p.MiniBatchSize = 100
	s := NewAdadeltaOptimizer()
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = 0.001

	testSolver(t, "Adadelta", s, p)
}

func TestAdamSolver(t *testing.T) {
	p := NewRandomLinearProblem(10000, 5, 5)
	s := NewAdamOptimizer()
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = .5
	testSolver(t, "AdamSolver", s, p)
}
