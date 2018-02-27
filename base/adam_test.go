package base

import (
	"testing"
)

func TestAdamSolver(t *testing.T) {
	p := NewRandomLinearProblem(10000, 5, 5)
	s := NewAdamSolver()
	// our problem is simplissime with a convex loss function so start with a big StepSize
	s.StepSize = .5
	testSolver(t, "AdamSolver", s, p)
}
