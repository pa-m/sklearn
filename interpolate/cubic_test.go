package interpolate

import (
	"flag"
	"os"
	"os/exec"

	"golang.org/x/exp/rand"

	"testing"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func BenchmarkCubicSpline(b *testing.B) {
	xs := make([]float64, 200)
	ys := make([]float64, len(xs))
	for i := range xs {
		xs[i] = float64(i) - 9.5
		ys[i] = rand.NormFloat64()
	}
	for i := 0; i < b.N; i++ {
		f := CubicSpline(xs, ys)
		for _, x := range xs {
			f(x)
		}
	}
}

var visualDebug = flag.Bool("visual", false, "show plots")

func ExampleCubicSpline() {
	xs := make([]float64, 20)
	ys := make([]float64, len(xs))
	for i := range xs {
		xs[i] = float64(i) - 9.5
		ys[i] = rand.NormFloat64()
	}
	f := CubicSpline(xs, ys)
	x2 := make([]float64, 220)
	y2 := make([]float64, 220)
	for i := range x2 {
		x2[i] = float64(i)/10 - 10.5
		y2[i] = f(x2[i])
	}

	p, err := plot.New()
	check(err)
	scatter, err := plotter.NewScatter(&xy{xs, ys})
	line, err := plotter.NewLine(&xy{x2, y2})
	p.Add(scatter, line)

	if *visualDebug {
		filename := "/tmp/spline.svg"
		os.Remove(filename)
		if err := p.Save(16*vg.Inch, 9*vg.Inch, filename); err != nil {
			panic(err)
		}
		_, err = os.Stat("/usr/bin/display")
		if err == nil {
			cmd := exec.Command("/usr/bin/display", "/tmp/spline.svg")
			cmd.Start()
		}
	}

	// Output:
}
