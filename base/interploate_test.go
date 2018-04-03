package base

import "fmt"

func ExampleInterp1d() {
	x := []float64{1, 2, 3, 4}
	y := []float64{-1, -2, 6, 8}
	f := Interp1d(x, y)
	ea := func(expected, actual float64) {
		if expected != actual {
			fmt.Printf("expected:%g actual:%g\n", expected, actual)
		}
	}
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
