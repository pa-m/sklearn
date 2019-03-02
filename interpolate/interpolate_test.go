package interpolate

import (
	"fmt"
	"golang.org/x/exp/rand"
	"sort"
	"testing"
)

func ExampleInterp1d() {
	x := []float64{1, 2, 3, 4}
	y := []float64{-1, -2, 6, 8}
	ea := func(expected, actual float64) {
		if expected != actual {
			fmt.Printf("expected:%g actual:%g\n", expected, actual)
		}
	}
	subtest := func(shuffle bool) {
		if shuffle {
			perm := rand.Perm(len(x))
			for sort.IntsAreSorted(perm) {
				perm = rand.Perm(len(x))
			}
			for i, p := range perm {
				if p > i {
					x[p], x[i] = x[i], x[p]
					y[p], y[i] = y[i], y[p]
				}
			}
		}
		f := Interp1d(x, y)
		ea(1, f(-1))
		ea(0, f(0))
		ea(-1, f(1))
		ea(-1.5, f(1.5))
		ea(-2, f(2))
		ea(2, f(2.5))
		ea(6, f(3))
		ea(7, f(3.5))
		ea(8, f(4))
		ea(10, f(5))
	}
	subtest(false)
	subtest(true)

	// Output:
}

func ExampleInterp2d() {
	x := []float64{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}
	y := []float64{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}
	z := []float64{0, 0, 0, 0, 0, 4, 6, 0, 0, 8, 10, 0, 0, 0, 0, 0}
	f := Interp2d(x, y, z)
	ea := func(expected, actual float64) {
		if expected != actual {
			fmt.Printf("expected:%g actual:%g\n", expected, actual)
		}
	}
	ea(7, f(2.5, 2.5))
	ea(5.5, f(2.25, 2.25))
	// Output:
}

func TestInterp2d(t *testing.T) {
	x := []float64{25, 25, 30, 30}
	y := []float64{5, 10, 5, 10}
	z := []float64{6.4, 7.1, 5.8, 6.4}
	f := Interp2d(x, y, z)
	ea := func(expected, actual float64) {
		if expected != actual {
			fmt.Printf("expected:%g actual:%g\n", expected, actual)
		}
	}
	cop25 := 6.4 + (8.-5.)/(10.-5.)*(7.1-6.4)
	cop30 := 5.8 + (8.-5.)/(10.-5.)*(6.4-5.8)
	cop := cop25 + (15-25)/(30-25)*(cop30-cop25)
	//fmt.Println("cop25:", cop25, "cop30:", cop30, "cop(15,8):", cop)
	ea(cop, f(15, 8))
}

// TODO v https://github.com/diazona/interp2d
