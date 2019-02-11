package svm

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func ExampleSVR() {
	/*
		https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
	*/

	// Generate sample data
	NSamples, NFeatures, NOutputs := 40, 1, 1
	X := mat.NewDense(NSamples, NFeatures, nil)
	Y := mat.NewDense(NSamples, NOutputs, nil)
	{
		rnd := rand.New(rand.NewSource(5))
		mX := X.RawMatrix()
		for sample := 0; sample < mX.Rows; sample++ {
			mX.Data[sample] = 5 * rnd.Float64()
		}
		sort.Float64s(mX.Data)
		mY := Y.RawMatrix()
		for sample := 0; sample < mY.Rows; sample++ {
			mY.Data[sample] = math.Sin(mX.Data[sample])
			if sample%5 == 0 {
				mY.Data[sample] += (0.5 - rnd.Float64())
			}
		}
	}
	// rescale in -1,1
	X, _ = preprocessing.NewMinMaxScaler([]float64{-1, 1}).FitTransform(X, nil)
	Epsilon := 0.05

	// # Fit regression model
	Ypred := map[string]*mat.Dense{}
	for _, opt := range []struct {
		kernel           string
		C, gamma, degree float64
	}{
		{kernel: "rbf", C: 1e3, gamma: .1},
		{kernel: "linear", C: 1e3},
		{kernel: "poly", C: 1e3, degree: 2},
	} {
		Ypred[opt.kernel] = &mat.Dense{}
		svr := NewSVR()
		svr.Kernel = opt.kernel
		svr.C = opt.C
		svr.Epsilon = Epsilon
		svr.Gamma = opt.gamma
		svr.Degree = opt.degree
		svr.MaxIter = 5
		svr.Fit(X, Y)
		svr.Predict(X, Ypred[opt.kernel])
		log.Println(base.MatStr(X, Y, Ypred[opt.kernel]))
	}

	if *visualDebug {
		// Look at the results
		pngfile := fmt.Sprintf("/tmp/ExampleSVR.png")
		os.Remove(pngfile)

		p, _ := plot.New()
		p.Title.Text = "Support vector regression"
		p.X.Label.Text = "data"
		p.Y.Label.Text = "target"
		xys := func(X, Y mat.Matrix, dy float64) (xy plotter.XYs) {
			imax, _ := Y.Dims()
			for i := 0; i < imax; i++ {
				xy = append(xy, struct{ X, Y float64 }{X.At(i, 0), Y.At(i, 0) + dy})

			}
			return
		}
		s, _ := plotter.NewScatter(xys(X, Y, 0))
		s.GlyphStyle.Shape = draw.CircleGlyph{}
		s.Color = color.RGBA{0xff, 0x80, 0x00, 0xFF}
		p.Add(s)
		p.Legend.Add("data", s)

		colors := map[string]color.Color{
			"rbf":    color.RGBA{0, 0, 0xff, 0xff},       //navy,
			"linear": color.RGBA{0, 0xff, 0xff, 0xff},    //cyan,
			"poly":   color.RGBA{0x64, 0x95, 0xed, 0xff}, //cornflower blue
		}
		labels := map[string]string{
			"rbf":    "RBF model",
			"linear": "Linear model",
			"poly":   "Polynomial model",
		}
		for kernel, Yp := range Ypred {
			// xys := func(X, Yp mat.Matrix) (xy plotter.XYs) {
			// 	imax, _ := X.Dims()
			// 	for i := 0; i < imax; i++ {
			// 		xy = append(xy, struct{ X, Y float64 }{X.At(i, 0), Yp.At(i, 0)})

			// 	}
			// 	return
			// }
			for _, dy := range []float64{-Epsilon, 0, Epsilon} {
				s, err := plotter.NewLine(xys(X, Yp, dy))
				if err != nil {
					panic(err)
				}
				s.Color = colors[kernel]
				if dy != 0 {
					s.LineStyle.Dashes = []vg.Length{3, 3}
				}
				p.Add(s)
				if dy == 0 {
					p.Legend.Add(labels[kernel], s)
				}
			}
		}

		if err := p.Save(6*vg.Inch, 4*vg.Inch, pngfile); err != nil {
			panic(err)
		}

		cmd := exec.Command("display", pngfile)
		err := cmd.Start()
		if err != nil {
			fmt.Println(err.Error())
		}
		time.Sleep(200 * time.Millisecond)
		os.Remove(pngfile)
	}
	// Output:
}
