package linearmodel

import (
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"os/exec"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var _ base.Predicter = &LogisticRegression{}
var visualDebug = flag.Bool("visual", false, "output images for benchmarks and test data")

func ExampleLogisticRegression() {

	// adapted from http://scikit-learn.org/stable/_downloads/plot_iris_logistic.ipynb
	ds := datasets.LoadIris()

	// we only take the first _ features.
	nSamples, _ := ds.X.Dims()
	X, YTrueClasses := ds.X.Slice(0, nSamples, 0, 2).(*mat.Dense), ds.Y
	h := .02 // step size in the mesh

	regr := NewLogisticRegression()
	regr.Alpha = 1e-5
	regr.beforeMinimize = func(problem optimize.Problem, initX []float64) {
		// check gradients
		settings := &fd.Settings{Step: 1e-8}
		gradFromModel := make([]float64, len(initX))
		gradFromFD := make([]float64, len(initX))
		problem.Func(initX)
		problem.Grad(gradFromModel, initX)
		fd.Gradient(gradFromFD, problem.Func, initX, settings)
		for i := range initX {
			if math.Abs(gradFromFD[i]-gradFromModel[i]) > 1e-4 {
				panic(fmt.Errorf("bad gradient, expected:\n%.3f\ngot:\n%.3f", gradFromFD, gradFromModel))
			}
		}
	}

	log.SetPrefix("ExampleLogisticRegression_Fit_iris:")
	defer log.SetPrefix("")

	// we create an instance of our Classifier and fit the data.
	regr.Fit(X, YTrueClasses)

	accuracy := regr.Score(X, YTrueClasses)
	if accuracy >= 0.833 {
		fmt.Println("ok")
	} else {
		fmt.Printf("Accuracy:%.3f\n", accuracy)
	}

	// Put the result into a color plot
	if *visualDebug {
		// Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
		var xmin, xmax = mat.Min(X.ColView(0)) - .5, mat.Max(X.ColView(0)) + .5

		var ymin, ymax = mat.Min(X.ColView(1)) - .5, mat.Max(X.ColView(1)) + .5

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
			return
		}
		var xx, yy = npmeshgrid(nparange(xmin, xmax, h), nparange(ymin, ymax, h))
		Xgrid := npc(xx, yy)
		Z := regr.Predict(Xgrid, nil)

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
		colors1 := []color.RGBA{{166, 206, 227, 255}, {253, 191, 111, 255}, {177, 89, 40, 255}}
		for cls := 0; cls <= 2; cls++ {
			s, _ := plotter.NewScatter(xys(Xgrid, Z, cls))
			s.GlyphStyle.Shape = draw.BoxGlyph{}
			s.GlyphStyle.Color = colors1[cls]
			s.GlyphStyle.Radius = 1
			plt.Add(s)

			s1, _ := plotter.NewScatter(xys(X, YTrueClasses, cls))
			s1.GlyphStyle.Shape = draw.CircleGlyph{}
			s1.GlyphStyle.Radius = 4
			s1.GlyphStyle.Color = colors1[cls]
			plt.Add(s1)
			plt.Legend.Add(ds.TargetNames[cls], s1)
		}
		plt.X.Label.Text = ds.FeatureNames[0]
		plt.Y.Label.Text = ds.FeatureNames[1]
		// Save the plot to a PNG file.
		pngfile := "/tmp/ExampleLogisticRegression.png"
		os.Remove(pngfile)
		if err := plt.Save(7*vg.Inch, 7*vg.Inch, pngfile); err != nil {
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
	// ok
}
