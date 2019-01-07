package linearmodel

import (
	"flag"
	"fmt"
	"image/color"
	"math"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var visualDebug = flag.Bool("visual", false, "output images for benchmarks and test data")

func TestLogRegExamScore(t *testing.T) {
	X, Ytrue := datasets.LoadExamScore()
	nSamples, nFeatures := X.Dims()
	preprocessing.AddDummyFeature(X)
	nFeatures++
	regr := NewLogisticRegression()
	regr.FitIntercept = false // Fintintercept=false because we already added ones columns instead

	Ybin := regr.EncodeLabels(Ytrue)
	_, NClasses := Ybin.Dims()
	// we allocate Coef here because we use it for loss and grad tests before Fit
	regr.Coef = mat.NewDense(nFeatures, NClasses, nil)

	// Alpha=0 here because following test values require no regularization
	regr.Alpha = 0.

	Ypred := &mat.Dense{}
	Ydiff := &mat.Dense{}
	grad := &mat.Dense{}

	J := math.Inf(1)
	loss := func() (J float64) {
		J = regr.LossFunction(Ybin, X, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction, true)
		return
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := []float64{grad.At(0, 0), grad.At(1, 0), grad.At(2, 0)}
		//fmt.Printf("%s grad=%v expected %v\n", context, actualGradient, expectedGradient)
		for j := 0; j < nFeatures; j++ {
			if !floats.EqualWithinAbs(expectedGradient[j], actualGradient[j], 1e-4) {
				t.Errorf("%s grad=%v expected %v", context, actualGradient, expectedGradient)
				return
			}
		}
	}
	chkTheta := func(context string, expectedTheta []float64) {
		actualTheta := []float64{regr.Coef.At(0, 0), regr.Coef.At(1, 0), regr.Coef.At(2, 0)}
		for j := 0; j < nFeatures; j++ {
			if !floats.EqualWithinAbs(expectedTheta[j], actualTheta[j], 1e-2) {
				t.Errorf("%s theta=%v expected %v", context, actualTheta, expectedTheta)
				return
			}
		}
	}
	J = loss()
	chkLoss("initial loss", 0.693)
	chkGrad("initial gradient", []float64{-0.1000, -12.0092, -11.2628})

	regr.Coef.SetCol(0, []float64{-24, 0.2, 0.2})
	J = loss()
	chkLoss("at test theta", 0.218)

	// test Fit with various gonum/optimize Method

	var GOMethodCreators = map[string]func() optimize.Method{
		"bfgs": func() optimize.Method { return &optimize.BFGS{} },
		"cg":   func() optimize.Method { return &optimize.CG{} },
		//&optimize.GradientDescent{},
		"lbfgs": func() optimize.Method { return &optimize.LBFGS{} },
		//&optimize.NelderMead{},
		//&optimize.Newton{}
	}
	printer := optimize.NewPrinter()
	//printer.HeadingInterval = 1
	//printer.ValueInterval = 0
	unused(printer)

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	for _, methodCreator := range GOMethodCreators {
		testSetup := fmt.Sprintf("(%T)", methodCreator())
		regr.Coef.SetCol(0, []float64{-24, 0.2, 0.2})
		regr.Options.GOMethodCreator = methodCreator
		//regr.Options.Recorder = printer
		start := time.Now()
		regr.Fit(X, Ybin)
		elapsed := time.Since(start)
		J = loss()

		chkLoss("after fit "+testSetup, 0.2030)
		chkTheta("after fit "+testSetup, []float64{-25.161, 0.206, 0.201})
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
	}

	// test Fit with various base.Optimizer

	var Optimizers = []string{"sgd", "adagrad", "rmsprop", "adadelta", "adam"}
	for _, optimizer := range Optimizers {
		testSetup := optimizer
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.SetCol(0, []float64{-24, 0.2, 0.2})
		}
		regr.Options.MiniBatchSize = nSamples
		regr.Solver = optimizer
		//regr.Options.Epochs = 1e5

		start := time.Now()
		regr.Fit(X, Ybin)
		elapsed := time.Since(start)
		J = loss()

		chkLoss("after fit "+testSetup, 0.2030)
		chkTheta("after fit "+testSetup, []float64{-25.161, 0.206, 0.201})
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
	}
	fmt.Println("LogisticRegression BEST SETUP:", best)
}

func TestLogRegMicrochipTest(t *testing.T) {
	X, Ytrue := datasets.LoadMicroChipTest()
	nSamples, nFeatures := X.Dims()

	//Xp, _ := preprocessing.NewPolynomialFeatures(6).Fit(X, Ytrue).Transform(X, Ytrue)
	// add poly features manually to have same order
	Xp := mat.NewDense(nSamples, 28, nil)
	c := 0
	for i := 0; i <= 6; i++ {
		for j := 0; j <= i; j++ {
			for s := 0; s < nSamples; s++ {
				Xp.Set(s, c, math.Pow(X.At(s, 0), float64(i-j))*math.Pow(X.At(s, 1), float64(j)))
			}
			c++
		}
	}

	_, nFeatures = Xp.Dims()
	_, nOutputs := Ytrue.Dims()
	regr := NewLogisticRegression()
	regr.FitIntercept = false // Fintintercept=false because we added polynomial features

	// we allocate Coef here because we use it for loss and grad tests before Fit
	regr.Coef = mat.NewDense(nFeatures, nOutputs, nil)

	// Alpha=1 here because following test requires it
	regr.Alpha = 1.
	regr.L1Ratio = 0.

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)
	grad := mat.NewDense(nFeatures, nOutputs, nil)

	J := math.Inf(1)
	loss := func() (J float64) {
		J = regr.LossFunction(Ytrue, Xp, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction, true)
		return
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := grad.RawRowView(0)[0:len(expectedGradient)]

		//fmt.Printf("%s grad=%v expected %v\n", context, actualGradient, expectedGradient)
		for j := 0; j < len(expectedGradient); j++ {
			if !floats.EqualWithinAbs(expectedGradient[j], actualGradient[j], 1e-4) {
				t.Errorf("%s grad=%v expected %v", context, actualGradient, expectedGradient)
				return
			}
		}
	}

	J = loss()
	chkLoss("Microchip initial loss", 0.693)
	chkGrad("Microchip initial gradient", []float64{0.0085, 0.0188, 0.0001, 0.0503, 0.0115})

	regr.Coef.Apply(func(j, o int, _ float64) float64 { return 1. }, regr.Coef)
	regr.Alpha = 10.

	J = regr.LossFunction(Ytrue, Xp, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction, true)
	chkLoss("At test theta", 3.164)
	chkGrad("at test theta", []float64{0.3460, 0.1614, 0.1948, 0.2269, 0.0922})

	// test Fit Microchip with various gonum/optimize Method

	var GOMethodCreators = map[string]func() optimize.Method{
		//"bfgs": func() optimize.Method { return &optimize.BFGS{} },
		//"cg":   func() optimize.Method { return &optimize.CG{} },
		//&optimize.GradientDescent{},
		"lbfgs": func() optimize.Method { return &optimize.LBFGS{} },
		//&optimize.NelderMead{},
		//&optimize.Newton{}
	}
	printer := optimize.NewPrinter()
	//printer.HeadingInterval = 1
	//printer.ValueInterval = 0
	unused(printer)

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	for _, methodCreator := range GOMethodCreators {
		testSetup := fmt.Sprintf("(%T)", methodCreator())
		regr.Options.GOMethodCreator = methodCreator
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.Sub(Theta, Theta)
		}
		regr.Alpha = 1.

		//regr.Options.Recorder = printer
		start := time.Now()
		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, true, nil)
		expectedAccuracy := 0.81
		if accuracy < expectedAccuracy {
			t.Errorf("%T accuracy=%g expected:%g", methodCreator(), accuracy, expectedAccuracy)
		}
	}

	// test Fit with various base.Optimizer
	var Optimizers = []string{
		//"sgd",
		//"adagrad",
		//"rmsprop",
		//"adadelta",
		"adam",
	}

	for _, optimizer := range Optimizers {
		testSetup := optimizer
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.Sub(Theta, Theta)
		}
		regr.Options.GOMethodCreator = nil
		//regr.Options.Recorder = printer
		regr.Options.MiniBatchSize = nSamples
		regr.Solver = optimizer
		regr.SolverConfigure = func(op base.Optimizer) {
			if optimizer == "sgd" {
				op.(*base.SGDOptimizer).StepSize = .25
			}
		}
		//regr.Options.Epochs = 1e5
		start := time.Now()
		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()

		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, true, nil)
		//fmt.Println("acc:", testSetup, accuracy)
		expectedAccuracy := 0.82
		if accuracy < expectedAccuracy {
			t.Errorf("%s accuracy=%g expected:%g", regr.Solver, accuracy, expectedAccuracy)
		}
	}
	fmt.Println("LogisticRegression BEST SETUP:", best)
}

func ExampleLogisticRegression() {
	// adapted from http://scikit-learn.org/stable/_downloads/plot_iris_logistic.ipynb
	ds := datasets.LoadIris()
	nSamples := len(ds.Target)
	// we only take the first two features.
	X, YTrueClasses := ds.X.Slice(0, nSamples, 0, 2).(*mat.Dense), ds.Y
	h := .02 // step size in the mesh
	_ = h
	logreg := NewLogisticRegression()
	C := 1e5
	logreg.Alpha = 1. / C

	// we create an instance of our Classifier and fit the data.
	logreg.Fit(X, YTrueClasses)

	Ypred := &mat.Dense{}
	logreg.Predict(X, Ypred)
	fmt.Printf("Accuracy:%.2f", metrics.AccuracyScore(YTrueClasses, Ypred, false, nil))

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
		Z := &mat.Dense{}
		logreg.Predict(Xgrid, Z)

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
	// Accuracy:93.00
}
