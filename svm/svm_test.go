package svm

import (
	"flag"
	"fmt"
	"image/color"
	"os"
	"os/exec"
	"time"

	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var visualDebug = flag.Bool("visual", false, "output images for benchmarks and test data")

func ExampleSVC() {
	// example adapted from http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
	X := mat.NewDense(16, 2, []float64{0.4, -0.7, -1.5, -1., -1.4, -0.9, -1.3, -1.2, -1.1, -0.2, -1.2,
		-0.4, -0.5, 1.2, -1.5, 2.1, 1., 1., 1.3, 0.8, 1.2, 0.5,
		0.2, -2., 0.5, -2.4, 0.2, -2.3, 0., -2.7, 1.3, 2.1})
	Y := mat.NewDense(16, 1, []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1})

	// rescale Y to -1..1
	yscaler := preprocessing.NewMinMaxScaler([]float64{-1, 1})
	X1 := X
	Y1, _ := yscaler.FitTransform(Y, nil)

	for _, kernel := range []string{
		"linear",
		"poly",
		"rbf",
	} {
		m := NewSVC()
		m.Kernel = kernel
		m.C = 1.
		m.MaxIter = 20
		m.Tol = 1.e-3
		m.Fit(X1, Y1)
		Ypred := mat.NewDense(16, 1, nil)
		m.Predict(X, Ypred)
		fmt.Printf("%s kernel, accuracy:%.3f\n", kernel, metrics.AccuracyScore(Y, Ypred, true, nil))
		// Put the result into a color plot
		if *visualDebug {
			// Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
			var xmin, xmax = -3., 3.

			var ymin, ymax = -3., 3.
			h := (ymax - ymin) / 100

			// xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
			nparange := func(min, max, h float64) []float64 {
				c := make([]float64, 0)
				for v := min; v <= max; v += h {
					c = append(c, v)
				}
				return c
			}
			npmeshgrid := func(xrange, yrange []float64) (xx, yy []float64) {
				for y := ymin; y <= ymax; y += h {
					for x := xmin; x <= xmax; x += h {
						xx = append(xx, x)
						yy = append(yy, y)
					}
				}
				return
			}
			npc := func(c ...[]float64) (XZ *mat.Dense) {
				XZ = mat.NewDense(len(c[0]), len(c), nil)
				for j, src := range c {
					XZ.SetCol(j, src)
				}
				return XZ
			}
			var xx, yy = npmeshgrid(nparange(xmin, xmax, h), nparange(ymin, ymax, h))
			Xgrid := npc(xx, yy)
			Z := &mat.Dense{}
			m.Predict(Xgrid, Z)
			plt, _ := plot.New()
			xys := func(X, Y mat.Matrix, cls int) (xy plotter.XYs) {
				imax, _ := Y.Dims()
				for i := 0; i < imax; i++ {
					if int(Y.At(i, 0)) == cls {
						xy = append(xy, struct{ X, Y float64 }{X.At(i, 0), X.At(i, 1)})
					}
				}
				return
			}
			colors1 := []color.RGBA{{166, 206, 227, 255}, {253, 191, 111, 255}}
			for cls := 0; cls <= 1; cls++ {
				s, _ := plotter.NewScatter(xys(Xgrid, Z, cls))
				s.GlyphStyle.Shape = draw.BoxGlyph{}
				s.GlyphStyle.Color = colors1[cls]
				s.GlyphStyle.Radius = 1
				plt.Add(s)

				s1, _ := plotter.NewScatter(xys(X, Y, cls))
				s1.GlyphStyle.Shape = draw.CircleGlyph{}
				s1.GlyphStyle.Radius = 4
				s1.GlyphStyle.Color = colors1[cls]
				plt.Add(s1)
				plt.Legend.Add(fmt.Sprintf("class %d", cls), s1)
			}
			plt.X.Label.Text = "X0"
			plt.Y.Label.Text = "X1"
			// Save the plot to a PNG file.
			pngfile := fmt.Sprintf("/tmp/ExampleSVC-%s.png", kernel)
			os.Remove(pngfile)
			if err := plt.Save(4*vg.Inch, 3*vg.Inch, pngfile); err != nil {
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

	}

	// Output:
	// linear kernel, accuracy:0.938
	// poly kernel, accuracy:1.000
	// rbf kernel, accuracy:1.000
}
