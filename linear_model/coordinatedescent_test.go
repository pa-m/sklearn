package linearModel

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"time"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func ExampleNewMultiTaskElasticNet() {
	clf := NewMultiTaskElasticNet()
	clf.Alpha = .1
	X, Y := mat.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3}), mat.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3})
	clf.fitcd(X, Y)
	fmt.Printf("%.8f\n", mat.Formatted(clf.Coef.T()))
	fmt.Printf("%.7f\n", mat.Formatted(clf.Intercept))
	Ypred := &mat.Dense{}
	clf.Predict(X, Ypred)
	fmt.Printf("Ypred:\n%.2f\n", mat.Formatted(Ypred))
	// Output:
	// [0.45663524  0.45612256]
	// [0.45663524  0.45612256]
	// [0.0872422  0.0872422]
}

func ExampleNewMultiTaskLasso() {
	clf := NewMultiTaskLasso()
	clf.Alpha = .1
	clf.fitcd(
		mat.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3}),
		mat.NewDense(3, 2, []float64{1, 1, 2, 2, 3, 3}),
	)
	fmt.Printf("%.8f\n", mat.Formatted(clf.Coef.T()))
	fmt.Printf("%.8f\n", mat.Formatted(clf.Intercept))
	// Output:
	// ⎡0.89393398  0.00000000⎤
	// ⎣0.89393398  0.00000000⎦
	// [0.10606602  0.10606602]
}

func ExampleNewLasso() {
	clf := NewLasso()
	clf.Alpha = .1
	clf.fitcd(
		mat.NewDense(3, 2, []float64{0, 0, 1, 1, 2, 2}),
		mat.NewDense(3, 1, []float64{0, 1, 2}),
	)
	fmt.Printf("%.8f\n", mat.Formatted(clf.Coef.T()))
	fmt.Printf("%.8f\n", mat.Formatted(clf.Intercept))
	// Output:
	// [0.85000000  0.00000000]
	// [0.15000000]
}

func ExampleNewElasticNet() {
	canPlot := false
	// adapted from http://scikit-learn.org/stable/_downloads/plot_train_error_vs_test_error.ipynb

	// Generate sample data
	//NSamplesTrain, NSamplesTest, NFeatures := 75, 150, 500
	NSamplesTrain, NSamplesTest, NFeatures := 75, 150, 500
	rand.Seed(0)
	coef := mat.NewDense(NFeatures, 1, nil)
	// only the top 10% features are impacting the model
	for feat := 0; feat < 50; feat++ {
		coef.Set(feat, 0, rand.NormFloat64())
	}
	X := mat.NewDense(NSamplesTrain+NSamplesTest, NFeatures, nil)
	{
		x := X.RawMatrix().Data
		for i := range x {
			x[i] = rand.NormFloat64()
		}

	}
	Y := &mat.Dense{}
	Y.Mul(X, coef)
	// Split train and test data
	rowslice := func(X mat.RawMatrixer, start, end int) *mat.Dense {
		rm := X.RawMatrix()
		return mat.NewDense(end-start, rm.Cols, rm.Data[start*rm.Stride:end*rm.Stride])
	}
	Xtrain, Xtest := rowslice(X, 0, NSamplesTrain), rowslice(X, NSamplesTrain, NSamplesTrain+NSamplesTest)
	Ytrain, Ytest := rowslice(Y, 0, NSamplesTrain), rowslice(Y, NSamplesTrain, NSamplesTrain+NSamplesTest)
	// Compute train and test errors
	//nalphas := 60
	nalphas := 20
	logalphas := make([]float64, nalphas)
	for i := range logalphas {
		logalphas[i] = -5 + 8*float64(i)/float64(nalphas)
	}
	trainErrors := make([]float64, nalphas)
	testErrors := make([]float64, nalphas)
	for ialpha, logalpha := range logalphas {
		//fmt.Println("ialpha=", ialpha)
		enet := NewElasticNet()
		enet.L1Ratio = 0.7
		enet.Alpha = math.Pow(10, logalpha)
		//enet.Tol = 1e-15
		//enet.Optimizer = base.NewAdadeltaOptimizer()
		//enet.Options.GOMethodCreator = func() optimize.Method { return &optimize.CG{} }
		enet.Fit(Xtrain, Ytrain)
		trainErrors[ialpha] = enet.Score(Xtrain, Ytrain)
		score := enet.Score(Xtest, Ytest)
		testErrors[ialpha] = score
	}
	iAlphaOptim := floats.MaxIdx(testErrors)
	alphaOptim := math.Pow(10, logalphas[iAlphaOptim])
	fmt.Printf("Optimal regularization parameter : %.6f", alphaOptim)
	//   # Plot outputs

	if canPlot {

		// plot result
		p, _ := plot.New()

		xys := func(X, Y []float64) plotter.XYs {
			var data plotter.XYs

			for i := range X {
				data = append(data, struct{ X, Y float64 }{X[i], Y[i]})
			}
			return data
		}
		s, _ := plotter.NewLine(xys(logalphas, trainErrors))
		s.Color = color.RGBA{0, 0, 255, 255}
		l, _ := plotter.NewLine(xys(logalphas, testErrors))
		l.Color = color.RGBA{255, 128, 0, 255}
		p.Add(s, l)
		p.Legend.Add("train", s)
		p.Legend.Add("test", l)

		// Save the plot to a PNG file.
		pngfile := "/tmp/elasticnet.png"
		os.Remove(pngfile)
		if err := p.Save(4*vg.Inch, 3*vg.Inch, pngfile); err != nil {
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
	// Optimal regularization parameter : 0.630957
}

func ExampleLasso_2() {
	// adapted from https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/ §4
	NSamples, NFeatures := 60, 15
	X, Y := mat.NewDense(NSamples, NFeatures, nil), mat.NewDense(60, 1, nil)
	for sample, i := 0, 60; i < 300; sample, i = sample+1, i+4 {
		X.Set(sample, 0, float64(i)*math.Pi/180.)
		Y.Set(sample, 0, math.Sin(X.At(sample, 0)) /*+rand.NormFloat64()*.15*/)
		//fmt.Printf("%d %.3f %.3f\t", i, X.At(sample, 0), Y.At(sample, 0))
	}
	//fmt.Println()
	for power := 2; power <= 15; power++ {
		v := &mat.VecDense{}
		v.ColViewOf(X, power-1)
		v.MulElemVec(X.ColView(0), X.ColView(power-2))
	}
	//fmt.Println(mat.Formatted(X))
	//fmt.Println(mat.Sum(Y.ColView(0)) / float64(NSamples))
	//fmt.Println(X.At(4, 14), Y.At(4, 0))
	m := NewLasso()
	m.FitIntercept = true
	m.Normalize = true
	m.Alpha = 1e-5
	m.L1Ratio = 1
	m.MaxIter = 1e5

	//m.Fit(X, Y)
	m.fitcd(X, Y)
	Ypred := &mat.Dense{}
	m.Predict(X, Ypred)
	rss := &mat.VecDense{}
	rss.SubVec(Ypred.ColView(0), Y.ColView(0))
	rss.MulElemVec(rss, rss)
	fmt.Printf("rss=%.3f %.2g %.2g\n", mat.Sum(rss), mat.Formatted(m.Intercept.T()), mat.Formatted(m.Coef.T()))
	// Output:
}
