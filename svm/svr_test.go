package svm

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"time"
	"unsafe"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// SVR struct
type SVR struct {
	BaseLibSVM
	Probability bool
	ClassWeight []float64
}

// NewSVR ...
// Kernel: "linear","poly","rbf","sigmoid" default is "rbf"
// if Gamma<=0 il will be changed to 1/NFeatures
// Cachesize is in MB. defaults to 200
func NewSVR() *SVR {
	m := &SVR{
		BaseLibSVM: BaseLibSVM{C: 1., Kernel: "rbf", Degree: 3., Gamma: 0., Coef0: 0., Shrinking: true, Tol: 1e-3, CacheSize: 200},
	}
	return m
}

// Clone for SVR
func (m *SVR) Clone() base.Transformer {
	clone := *m
	return &clone
}

func svrTrain(X *mat.Dense, Y []float64, C float64, KernelFunction func(X1, X2 []float64) float64, Tol float64, MaxPasses int, CacheSize uint) *Model {
	m, n := X.Dims()
	alphas := make([]float64, m, m)
	b := 0.
	E := make([]float64, m, m)
	eta := 0.
	passes := 0
	L, H := 0., 0.
	type KcacheEntry struct {
		i, j int
		v    float64
	}
	KcacheEntrySize := uint(unsafe.Sizeof(KcacheEntry{}))

	Kcache := make([]KcacheEntry, 0, CacheSize<<20/KcacheEntrySize)

	K := func(i, j int) float64 {
		if i > j {
			i, j = j, i
		}
		for off, e := range Kcache {
			if e.i == i && e.j == j {
				copy(Kcache[1:off+1], Kcache[0:off])
				Kcache[0] = e
				return e.v
			}
		}
		if uint(len(Kcache))*KcacheEntrySize >= CacheSize {
			Kcache = Kcache[0 : CacheSize/KcacheEntrySize]
		}
		e := KcacheEntry{i, j, KernelFunction(X.RawRowView(i), X.RawRowView(j))}
		Kcache = append([]KcacheEntry{e}, Kcache...)
		return e.v
	}
	calcE := func(i int) {
		//Calculate Ei = f(x(i)) - y(i) using E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
		sumAlphaYK := 0.
		for i1 := 0; i1 < m; i1++ {
			sumAlphaYK += alphas[i1] * K(i1, i)
		}
		E[i] = b + sumAlphaYK - Y[i]
	}
	for passes < MaxPasses {
		numChangedAlphas := 0
		// Step 1 Find a Lagrange multiplier α 1 {that violates the Karush–Kuhn–Tucker (KKT) conditions for the optimization problem.
		for i := 0; i < m; i++ {
			calcE(i)
			if (Y[i]*E[i] < -Tol && alphas[i] < C) || (Y[i]*E[i] > Tol && alphas[i] > 0) {
				// Step 2 Pick a second multiplier α 2  and optimize the pair ( α 1 , α 2 )
				// % In practice, there are many heuristics one can use to select
				// % the i and j. In this simplified code, we select them randomly.
				j := rand.Intn(m - 1)
				if j >= i {
					j++
				}
				calcE(j)

				alphaiold, alphajold := alphas[i], alphas[j]
				//% Compute L and H by (10) or (11).
				if Y[i] == Y[j] {
					L, H = math.Max(0, alphas[j]+alphas[i]-C), math.Min(C, alphas[j]+alphas[i])
				} else {
					L, H = math.Max(0, alphas[j]-alphas[i]), math.Min(C, C+alphas[j]-alphas[i])
				}
				if L == H {
					continue
				}
				Kij := K(i, j)
				Kii := K(i, i)
				Kjj := K(j, j)
				eta = 2*Kij - Kii - Kjj
				if eta >= 0 {
					continue
				}
				// % Compute and clip new value for alpha j using (12) and (15).
				alphas[j] = alphas[j] - (E[i]-E[j])/eta

				// % Clip
				alphas[j] = math.Max(L, math.Min(H, alphas[j]))
				// % Check if change in alpha is significant
				if alphas[j]-alphajold < Tol {
					alphas[j] = alphajold
					continue
				}
				// % Determine value for alpha i using (16).
				alphas[i] += (alphajold - alphas[j])
				// % Compute b1 and b2 using (17) and (18) respectively.
				b1 := b - E[i] - (alphas[i]-alphaiold)*Kij - (alphas[j]-alphajold)*Kij
				b2 := b - E[j] - (alphas[i]-alphaiold)*Kij - (alphas[j]-alphajold)*Kjj
				// % Compute b by (19).
				if 0 < alphas[i] && alphas[i] < C {
					b = b1
				} else if 0 < alphas[j] && alphas[j] < C {
					b = b2
				} else {
					b = (b1 + b2) / 2
				}
				// {
				// 	Eold := E[i]
				// 	calcE(i)
				// 	fmt.Printf("passes %d alpha[%d] %g => %g E[%d] %g=>%g\n", passes, i, alphaiold, alphas[i], i, Eold, E[i])
				// }
				numChangedAlphas++
			}
			// Step 3: Repeat steps 1 and 2 until convergence.
		}
		if numChangedAlphas == 0 {
			passes++
		} else {
			passes = 0
		}
	}
	idx := make([]int, 0)
	for i := 0; i < m; i++ {
		if alphas[i] > Tol {
			idx = append(idx, i)
		}
	}
	model := &Model{
		X:              mat.NewDense(len(idx), n, nil),
		Y:              make([]float64, len(idx)),
		KernelFunction: KernelFunction,
		B:              b,
		Alphas:         make([]float64, len(idx)),
		W:              make([]float64, n, n),
		Support:        idx,
	}
	for ii, i := range idx {
		model.X.SetRow(ii, X.RawRowView(i))
		model.Y[ii] = Y[i]
		model.Alphas[ii] = alphas[i]
	}
	for j := 0; j < n; j++ {
		for _, i := range idx {
			model.W[j] += alphas[i] * X.At(i, j)
		}
	}
	return model
}

// Fit for SVR
func (m *SVR) Fit(X, Y *mat.Dense) base.Transformer {
	m.BaseLibSVM.fit(X, Y, svrTrain)
	return m
}

func svrPredict(model *Model, X, Y *mat.Dense, output int) {
	NSamples, _ := X.Dims()

	Ymat := Y.RawMatrix()
	for i, yoff := 0, output; i < NSamples; i, yoff = i+1, yoff+Ymat.Stride {
		prediction := 0.
		for j := range model.Alphas {
			prediction += model.Alphas[j] * model.KernelFunction(X.RawRowView(i), model.X.RawRowView(j))
		}
		prediction += model.B
		Ymat.Data[yoff] = prediction
	}
	return
}

// Predict for SVR
func (m *SVR) Predict(X, Y *mat.Dense) base.Transformer {
	_, NOutputs := Y.Dims()
	if NOutputs == 0 {
		NSamples, _ := X.Dims()
		NOutputs = len(m.Model)
		*Y = *mat.NewDense(NSamples, NOutputs, nil)
	}
	base.Parallelize(-1, NOutputs, func(th, start, end int) {

		for output := start; output < end; output++ {

			svrPredict(m.Model[output], X, Y, output)

		}
	})
	return m
}

// Transform for SVR for pipeline
func (m *SVR) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, _ := X.Dims()
	Xout = X
	Yout = mat.NewDense(NSamples, len(m.Model), nil)
	m.Predict(X, Yout)
	return
}

func _ExampleSVR() {
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
				mY.Data[sample] += 3 * (0.5 - rnd.Float64())
			}
		}
	}
	// rescale in -1,1
	X, _ = preprocessing.NewMinMaxScaler([]float64{-1, 1}).FitTransform(X, nil)
	//Y, _ = preprocessing.NewMinMaxScaler([]float64{-1, 1}).FitTransform(Y, nil)

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
		svr.Gamma = opt.gamma
		svr.Degree = opt.degree
		svr.MaxIter = 100
		svr.Fit(X, Y)
		svr.Predict(X, Ypred[opt.kernel])
	}

	if *visualDebug {
		// Look at the results
		pngfile := fmt.Sprintf("/tmp/ExampleSVR.png")
		os.Remove(pngfile)

		p, _ := plot.New()
		p.Title.Text = "Support vector regression"
		p.X.Label.Text = "data"
		p.Y.Label.Text = "target"
		xys := func(X, Y mat.Matrix) (xy plotter.XYs) {
			imax, _ := Y.Dims()
			for i := 0; i < imax; i++ {
				xy = append(xy, struct{ X, Y float64 }{X.At(i, 0), Y.At(i, 0)})

			}
			return
		}
		s, _ := plotter.NewScatter(xys(X, Y))
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
			xys := func(X, Yp mat.Matrix) (xy plotter.XYs) {
				imax, _ := X.Dims()
				for i := 0; i < imax; i++ {
					xy = append(xy, struct{ X, Y float64 }{X.At(i, 0), Yp.At(i, 0)})

				}
				return
			}

			s, err := plotter.NewLine(xys(X, Yp))
			if err != nil {
				panic(err)
			}
			s.Color = colors[kernel]
			p.Add(s)
			p.Legend.Add(labels[kernel], s)

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
