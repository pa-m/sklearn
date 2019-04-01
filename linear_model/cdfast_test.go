package linearmodel

import (
	"fmt"
	"testing"

	"github.com/pa-m/sklearn/base"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
)

func TestEnetCoordinateDescent(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	w := mat.NewVecDense(3, []float64{1, 2, 3})
	Y := &mat.VecDense{}
	Y.MulVec(X, w)
	alpha := .001
	beta := .001
	maxIter := 1500
	tol := 1e-6
	rng := rand.New(base.NewLockedSource(0))
	random := false
	positive := false
	enetCoordinateDescent(w, alpha, beta, X, Y, maxIter, tol, rng, random, positive)
	fmt.Printf("%.3f\n", mat.Formatted(w.T()))
}
