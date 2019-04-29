package neuralnetwork

import (
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/chewxy/math32"
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	modelselection "github.com/pa-m/sklearn/model_selection"
	"github.com/pa-m/sklearn/pipeline"
	"golang.org/x/exp/rand"

	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var _ = []base.Predicter{&MLPRegressor{}, &MLPClassifier{}}

var visualDebug = flag.Bool("visual", false, "show plots")

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func TestMLPClassifierMicrochip(t *testing.T) {
	X, Ytrue := datasets.LoadMicroChipTest()
	nSamples, _ := X.Dims()

	// add polynomial features
	poly := preprocessing.NewPolynomialFeatures(6)
	poly.IncludeBias = false
	poly.Fit(X, nil)
	Xp, _ := poly.Transform(X, nil)

	_, nFeatures := Xp.Dims()
	_, nOutputs := Ytrue.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)

	Alpha := 1.
	mlp := NewMLPClassifier([]int{}, "logistic", "adam", Alpha)
	mlp.BatchSize = nSamples

	// we allocate Coef here because we use it for loss and grad tests before Fit
	mlp.initialize(Ytrue.RawMatrix().Cols, []int{nFeatures, nOutputs}, true, false)
	mlp.WarmStart = true
	mlp.Shuffle = false

	var J float64
	loss := func() float64 {
		mlp.MaxIter = 1
		mlp.Fit(Xp, Ytrue)
		return mlp.Loss
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := mlp.packedGrads[:len(expectedGradient)]

		//fmt.Printf("%s grad=%v expected %v\n", context, actualGradient, expectedGradient)
		for j := 0; j < len(expectedGradient); j++ {
			if !floats.EqualWithinAbs(expectedGradient[j], actualGradient[j], 1e-4) {
				t.Errorf("%s grad=%v expected %v", context, actualGradient, expectedGradient)
				return
			}
		}
	}
	t.Run("loss and grad with Alpha=0", func(t *testing.T) {
		for i := range mlp.packedParameters {
			mlp.packedParameters[i] = 0
		}
		mlp.Alpha = 1
		J = loss()
		chkLoss("Microchip initial loss", 0.693)
		chkGrad("Microchip initial gradient", []float64{0.0085, 0.0188, 0.0001, 0.0503, 0.0115})
		for i := range mlp.packedParameters {
			mlp.packedParameters[i] = 1
		}

	})
	t.Run("loss and grad with Alpha=10", func(t *testing.T) {
		mlp.Alpha = 10.

		J = loss()
		chkLoss("At test theta", 3.164)
		chkGrad("at test theta", []float64{0.3460, 0.1614, 0.1948, 0.2269, 0.0922})
	})
	// try different solvers

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	// // test Fit with various base.Optimizer
	var Optimizers = []string{
		"sgd",
		// "adagrad",
		// "rmsprop",
		// "adadelta",
		"adam",
		"lbfgs",
	}

	checkGradients := func(problem optimize.Problem, initX []float64) {
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

	for _, optimizer := range Optimizers {
		t.Run(optimizer, func(t *testing.T) {

			testSetup := optimizer
			mlp := NewMLPClassifier([]int{}, "logistic", optimizer, 1)
			mlp.RandomState = base.NewLockedSource(1)
			mlp.initialize(Ytrue.RawMatrix().Cols, []int{nFeatures, nOutputs}, true, false)
			for i := range mlp.packedParameters {
				mlp.packedParameters[i] = 0
			}
			mlp.WarmStart = true
			mlp.MaxIter = 400
			mlp.LearningRateInit = .11
			mlp.BatchSize = 118 //1,2,59,118
			mlp.beforeMinimize = checkGradients

			start := time.Now()
			mlp.Fit(Xp, Ytrue)
			elapsed := time.Since(start)
			J := mlp.Loss

			if J < bestLoss {
				bestLoss = J
				best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
			}
			if elapsed < bestTime {
				bestTime = elapsed
				best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
			}
			mlp.Predict(Xp, Ypred)
			accuracy := metrics.AccuracyScore(Ytrue, Ypred, true, nil)
			// accuracy should be over 0.83
			expectedAccuracy := 0.8305
			if accuracy < expectedAccuracy {
				t.Errorf("%s accuracy=%.3g expected:%.3g", optimizer, accuracy, expectedAccuracy)
			}
		})
	}
	fmt.Println("MLPClassifier BEST SETUP:", best)

	// // fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred,true,nil))
	// fmt.Println("ok")
}

func ExampleMLPClassifier_Unmarshal() {
	// json saved from scikit learn using:
	// dic=mlp.get_params(True)
	// for k in ['out_activation_']:
	//   dic[k] = getattr(mlp,k)
	// for k in ['intercepts_','coefs_']:
	//   dic[k] = [x.tolist() for x in getattr(mlp,k)]
	// json.dumps(dic)
	buf := []byte(`{"activation": "logistic", "alpha": 0.0001, "batch_size": "auto", "beta_1": 0.9, "beta_2": 0.999, "early_stopping": false, "epsilon": 1e-08, "hidden_layer_sizes": [], "learning_rate": "constant", "learning_rate_init": 0.001, "max_iter": 400, "momentum": 0.9, "n_iter_no_change": 10, "nesterovs_momentum": true, "power_t": 0.5, "random_state": 7, "shuffle": true, "solver": "adam", "tol": 0.0001, "validation_fraction": 0.1, "verbose": false, "warm_start": false, "out_activation_": "logistic", "intercepts_": [[0.5082271055138958]], "coefs_": [[[-0.18963335144967644], [0.2744326667319166], [-0.0068960058868800505], [-0.1870170339590578], [0.33640123639043934], [0.14343164310877599], [-0.2840940844068544], [-0.06035740527894848], [-0.015548157556294752], [-0.09766841821748058], [-0.13516966516561582], [0.01180873002271984], [-0.37004002347719184], [-0.3146740174229507], [-0.010236340304847167], [0.034725564039145625], [0.07596312959511524], [0.07031424991074327], [0.03226286238715042], [-0.11777688776136522], [-0.0862585580460505], [0.046039278168215306], [-0.32297687193126345], [0.004283074654547827], [0.013040383833634088], [-0.047491825368820184], [-0.12259098577236986]]]}`)
	mlp := NewMLPClassifier([]int{}, "", "", 0)
	err := mlp.Unmarshal(buf)
	if err != nil {
		panic(err)
	}

	X, Ytrue := datasets.LoadMicroChipTest()
	nSamples, _ := X.Dims()
	_ = nSamples
	X, _ = preprocessing.NewStandardScaler().FitTransform(X, nil)
	// add polynomial features
	pf := preprocessing.NewPolynomialFeatures(6)
	pf.IncludeBias = false
	pf.Fit(X, Ytrue)
	Xp, _ := pf.Transform(X, Ytrue)
	Ypred := mat.NewDense(nSamples, 1, nil)
	// reset OutActivation because it's not in params
	// mlp.OutActivation = "logistic"
	mlp.Predict(Xp, Ypred)
	accuracy := metrics.AccuracyScore(Ytrue, Ypred, true, nil)
	if accuracy > .83 {
		fmt.Println("ok")
	} else {
		fmt.Println("accuracy", accuracy)
		fmt.Println("Ytrue", Ytrue.RawMatrix().Data)
		fmt.Println("Ypred", Ypred.RawMatrix().Data)
	}
	// Output:
	// ok
}

func ExampleMLPClassifier_Fit_mnist() {
	// fitting mnist with randomstate 7, shuffle, batchnorm,400 iterations should allow accuracy 99.96%. use embedded label binarizer

	MaxIter := 400
	expectedMinAccuracy := .999
	if testing.Short() {
		log.Println("ExampleMLPClassifier_Fit_mnist reducted iterations because testing with -short")
		MaxIter = 40
		expectedMinAccuracy = .95

	}
	X, Y := datasets.LoadMnist()
	mlp := NewMLPClassifier([]int{25}, "logistic", "adam", 0)
	mlp.RandomState = base.NewLockedSource(7)
	mlp.Shuffle = true
	mlp.BatchNormalize = true
	mlp.MaxIter = MaxIter

	log.SetPrefix("ExampleMLPClassifier_Fit_mnist:")
	defer log.SetPrefix("")

	mlp.Fit(X, Y)
	pred := mlp.Predict(X, nil)
	acc := metrics.AccuracyScore(Y, pred, true, nil)
	if acc < expectedMinAccuracy {
		fmt.Printf("Accuracy:%.2f%%\n", acc*100)
	} else {
		fmt.Println("ok")
	}

	// Output:
	// ok
}

func ExampleMLPClassifier_Predict_mnist() {
	X, Y := datasets.LoadMnist()
	lb := preprocessing.NewLabelBinarizer(0, 1)
	X, Ybin := lb.FitTransform(X, Y)
	Theta1T, Theta2T := datasets.LoadMnistWeights()
	mlp := NewMLPClassifier([]int{25}, "logistic", "adam", 0)
	mlp.Shuffle = false
	mlp.initialize(Ybin.RawMatrix().Cols, []int{400, 25, 10}, true, true)
	mat.NewDense(401, 25, mlp.packedParameters[:401*25]).Copy(Theta1T.T())
	mat.NewDense(26, 10, mlp.packedParameters[401*25:]).Copy(Theta2T.T())
	mlp.WarmStart = true

	predBin := mat.NewDense(Ybin.RawMatrix().Rows, Ybin.RawMatrix().Cols, nil)
	mlp.Predict(X, predBin)
	//_, pred := lb.InverseTransform(nil, predBin)
	acc := metrics.AccuracyScore(Ybin, predBin, true, nil)
	fmt.Printf("Accuracy:%.2f%%\n", acc*100)

	// Output:
	// Accuracy:97.52%
}

func Benchmark_Fit_mnist(b *testing.B) {
	// (cd exp && generate.sh)
	// go test ./neural_network -run Benchmark_Fit_Mnist -bench ^Benchmark_Fit_Mnist -cpuprofile /tmp/cpu.prof -memprofile /tmp/mem.prof -benchmem
	// go test ./exp -run BenchmarkMnist -bench ^Benchmark_Fit_Mnist -cpuprofile /tmp/cpu.prof -memprofile /tmp/mem.prof -benchmem
	// go tool pprof /tmp/cpu.prof
	X, Y := datasets.LoadMnist()

	X, Ybin := (&preprocessing.LabelBinarizer{}).FitTransform(X, Y)
	Theta1, Theta2 := datasets.LoadMnistWeights()
	_, _ = Theta1, Theta2
	mlp := NewMLPClassifier([]int{25}, "logistic", "adam", 0.)
	mlp.BatchSize = 5000
	mlp.Shuffle = false
	_, NFeatures := X.Dims()
	_, NOutputs := Ybin.Dims()
	mlp.initialize(Ybin.RawMatrix().Cols, []int{NFeatures, 25, NOutputs}, true, true)
	mat.NewDense(401, 25, mlp.packedParameters[:401*25]).Copy(Theta1.T())
	mat.NewDense(26, 10, mlp.packedParameters[401*25:]).Copy(Theta2.T())
	mlp.WarmStart = true
	//J:=mlp.Loss
	b.ResetTimer()
	fmt.Println("Benchmark_Fit_Mnist b.N", b.N)
	mlp.MaxIter = b.N
	mlp.Fit(X, Ybin)
	fmt.Println("Benchmark_Fit_Mnist J", mlp.Loss)
}

//go test ./neural_network -run Benchmark_Fit_Mnist -bench ^Benchmark_Fit_Mnist -cpuprofile /tmp/cpu.prof -memprofile /tmp/mem.prof -benchmem
//BenchmarkMnist-12            100          17387518 ns/op           89095 B/op         30 allocs/op

func ExampleMLPClassifier_Fit_breast_cancer() {
	ds := datasets.LoadBreastCancer()

	scaler := preprocessing.NewStandardScaler()
	scaler.Fit(ds.X, ds.Y)
	X0, Y0 := scaler.Transform(ds.X, ds.Y)
	nSamples, _ := Y0.Dims()
	pca := preprocessing.NewPCA()
	pca.Fit(X0, Y0)
	X1, Y1 := pca.Transform(X0, Y0)
	thres := .995
	ExplainedVarianceRatio := 0.
	var nComponents int
	for nComponents = 0; nComponents < len(pca.ExplainedVarianceRatio) && ExplainedVarianceRatio < thres; nComponents++ {
		ExplainedVarianceRatio += pca.ExplainedVarianceRatio[nComponents]
	}
	fmt.Printf("ExplainedVarianceRatio %.3f %.3f\n", ExplainedVarianceRatio, pca.ExplainedVarianceRatio[0:nComponents])
	fmt.Printf("%d components explain %.2f%% of variance\n", nComponents, thres*100.)
	X1 = X1.Slice(0, nSamples, 0, nComponents).(*mat.Dense)
	poly := preprocessing.NewPolynomialFeatures(2)
	poly.IncludeBias = false

	poly.Fit(X1, Y1)
	X2, Y2 := poly.Transform(X1, Y1)

	m := NewMLPClassifier([]int{}, "logistic", "adam", 0.)
	m.RandomState = base.NewLockedSource(1)
	m.LearningRateInit = .02
	m.WeightDecay = .001
	m.MaxIter = 300

	log.SetPrefix("ExampleMLPClassifier_Fit_breast_cancer:")
	defer log.SetPrefix("")
	m.Fit(X2, Y2)
	accuracy := m.Score(X2, Y2)
	if accuracy <= .999 {
		fmt.Printf("accuracy:%.9f\n", accuracy)
	} else {
		fmt.Println("accuracy>0.999 ? true")
	}

	// Output:
	// ExplainedVarianceRatio 0.996 [0.443 0.190 0.094 0.066 0.055 0.040 0.023 0.016 0.014 0.012 0.010 0.009 0.008 0.005 0.003 0.003 0.002 0.002 0.002 0.001]
	// 20 components explain 99.50% of variance
	// accuracy>0.999 ? true

}

func ExampleMLPRegressor_Fit_boston() {
	// exmaple inspired from # https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
	// with wider_model
	// added weight decay and reduced epochs from 100 to 20
	ds := datasets.LoadBoston()
	X, Y := ds.X, ds.Y

	mlp := NewMLPRegressor([]int{20}, "relu", "adam", 0)
	mlp.RandomState = base.NewLockedSource(1)
	mlp.LearningRateInit = .05
	mlp.WeightDecay = .01
	mlp.Shuffle = false
	mlp.BatchSize = 5
	mlp.MaxIter = 100
	m := pipeline.MakePipeline(preprocessing.NewStandardScaler(), mlp)
	_ = m
	randomState := rand.New(base.NewLockedSource(7))
	scorer := func(Y, Ypred mat.Matrix) float64 {
		e := metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
		return e
	}
	mean := func(x []float64) float64 { return floats.Sum(x) / float64(len(x)) }

	log.SetPrefix("ExampleMLPRegressor_Fit_boston:")
	defer log.SetPrefix("")
	res := modelselection.CrossValidate(m, X, Y,
		nil,
		scorer,
		&modelselection.KFold{NSplits: 10, Shuffle: true, RandomState: randomState}, 10)
	fmt.Println(math.Sqrt(mean(res.TestScore)) < 20)

	// Output:
	// true
}

func Test_r2Score32(t *testing.T) {
	//1st example of sklearn metrics r2Score
	yTrue := blas32.General{Rows: 4, Cols: 1, Stride: 1, Data: []float32{3, -0.5, 2, 7}}
	yPred := blas32.General{Rows: 4, Cols: 1, Stride: 1, Data: []float32{2.5, 0.0, 2, 8}}
	r2Score := r2Score32(yTrue, yPred)
	eps := float32(1e-3)
	if math32.Abs(0.948-r2Score) > eps {
		t.Errorf("expected 0.948 got %g", r2Score)
	}

	yTrue = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{1, 2, 3}}
	yPred = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{1, 2, 3}}
	if math32.Abs(1.-r2Score32(yTrue, yPred)) >= 1e-3 {
		t.Error("expected 1")
	}
	yTrue = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{1, 2, 3}}
	yPred = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{2, 2, 2}}
	if math32.Abs(0.-r2Score32(yTrue, yPred)) >= 1e-3 {
		t.Error("expected 0")
	}
	yTrue = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{1, 2, 3}}
	yPred = blas32.General{Rows: 3, Cols: 1, Stride: 1, Data: []float32{3, 2, 1}}
	if math32.Abs(-3.-r2Score32(yTrue, yPred)) >= 1e-3 {
		t.Error("expected -3")
	}
}

func Test_accuracyScore32(t *testing.T) {
	// adapted from example in https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/metrics/classification.py
	Ypred, Ytrue := blas32.General{Rows: 4, Cols: 1, Stride: 1, Data: []float32{0, 2, 1, 3}}, blas32.General{Rows: 4, Cols: 1, Stride: 1, Data: []float32{0, 1, 2, 3}}
	expected, actual := float32(0.5), accuracyScore32(Ytrue, Ypred)
	if actual != expected {
		t.Errorf("expected %g, got %g", expected, actual)
	}
	Ypred, Ytrue = blas32.General{Rows: 2, Cols: 2, Stride: 2, Data: []float32{0, 1, 1, 1}}, blas32.General{Rows: 2, Cols: 2, Stride: 2, Data: []float32{1, 1, 1, 1}}
	expected, actual = float32(0.5), accuracyScore32(Ytrue, Ypred)
	if actual != expected {
		t.Errorf("expected %g, got %g", expected, actual)
	}
}

func TestMLPRegressor(t *testing.T) {
	mlp := NewMLPRegressor([]int{}, "relu", "adam", 0)
	mlp = mlp.PredicterClone().(*MLPRegressor) // for coverage
	mlp.IsClassifier()                         // for coverage
	X, Y, _ := datasets.MakeRegression(map[string]interface{}{"n_samples": 100, "n_features": 2})
	mlp.LearningRateInit = .1
	mlp.Fit(X, Y)
	if mlp.Score(X, Y) < .95 {
		t.Fail()
	}

}

func ExampleMLPClassifier_Fit_iris() {

	// adapted from http://scikit-learn.org/stable/_downloads/plot_iris_logistic.ipynb
	ds := datasets.LoadIris()

	// we only take the first _ features.
	nSamples, _ := ds.X.Dims()
	X, YTrueClasses := ds.X.Slice(0, nSamples, 0, 2).(*mat.Dense), ds.Y
	h := .02 // step size in the mesh

	mlp := NewMLPClassifier([]int{}, "logistic", "lbfgs", 1e-5)

	log.SetPrefix("ExampleMLPClassifier_Fit_iris:")
	defer log.SetPrefix("")

	// we create an instance of our Classifier and fit the data.
	mlp.Fit(X, YTrueClasses)

	accuracy := mlp.Score(X, YTrueClasses)
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
		Z := &mat.Dense{}
		mlp.Predict(Xgrid, Z)

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
		pngfile := "/tmp/ExampleMLPClassifier_Fit_iris.png"
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
