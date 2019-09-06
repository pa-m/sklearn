package gaussianprocess

import (
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/gaussian_process/kernels"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

var _ base.Predicter = &Regressor{}

func TestRegressor_LogMarginalLikelihood(t *testing.T) {
	//from plot_gpr_noisy.ipynb
	//# Instantiate a Gaussian Process model
	//kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
	//gp = Regressor(kernel=kernel, n_restarts_optimizer=9)
	kernel := &kernels.Product{KernelOperator: kernels.KernelOperator{
		K1: &kernels.ConstantKernel{ConstantValue: 1, ConstantValueBounds: [2]float64{1e-3, 1e3}},
		K2: &kernels.RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}},
	}}

	//# now the noisy case
	gp := NewRegressor(kernel)
	X := mat.NewDense(20, 1, nil)
	//X = np.linspace(0.1, 9.9, 20)
	{
		x := X.RawMatrix().Data
		for i := range x {
			x[i] = .1 + ((9.9 - .1) * float64(i) / (float64(len(x) - 1)))
		}
	}
	//np.random.seed(1)
	randomState := base.NewSource(uint64(1))
	//dy = 0.5 + 1.0 * np.random.random(y.shape)
	dy := make([]float64, 20)
	{
		for i := range dy {
			dy[i] = 0.5 + 1.0*randomState.Float64()
		}
	}
	noise := make([]float64, 20)
	{
		for i := range noise {
			noise[i] = randomState.NormFloat64() * dy[i]
		}
	}
	Y := mat.NewDense(20, 1, nil)
	{
		x := X.RawMatrix().Data
		y := Y.RawMatrix().Data
		for i := range y {
			y[i] = x[i]*math.Sin(x[i]) + noise[i]
		}
	}
	gp.Xtrain = X
	gp.Ytrain = Y

	expectedTheta := []float64{0., 2.30258509}
	actualTheta := mat.DenseCopyOf(gp.Theta()).RawMatrix().Data
	tol := 1e-3
	if !floats.EqualApprox(expectedTheta, actualTheta, tol) {
		t.Errorf("expected theta %g, got %g", expectedTheta, actualTheta)
	}

	gp.Alpha = []float64{0.840929357108728, 1.48919186929486, 0.250114387898944,
		0.643737557106026, 0.418293182306637, 0.350865010852674,
		0.470953077720126, 0.714972943117568, 0.804191902838055,
		1.079140206845401, 0.844918555309109, 1.404745264120745,
		0.496252972151796, 1.899207668484751, 0.278137673459101,
		1.369994192383228, 0.841448100445794, 1.120824152854495,
		0.410095431123376, 0.487345689062525}
	lml, grad := gp.LogMarginalLikelihood(gp.Theta(), true)
	expectedLml := -200.9855428488616
	if math.Abs(expectedLml-lml) > tol {
		t.Errorf("expected lml %g, got %g", expectedLml, lml)
	}
	expectedGrad := []float64{-0.07469052, -1.52187805}
	if !floats.EqualApprox(expectedGrad, grad, tol) {
		t.Errorf("expected grad %g, got %g", expectedGrad, grad)
	}
}

func TestRegressor_Predict(t *testing.T) {
	//gp = Regressor(kernel=kernel, n_restarts_optimizer=9)
	kernel := &kernels.Product{KernelOperator: kernels.KernelOperator{
		K1: &kernels.ConstantKernel{ConstantValue: 1, ConstantValueBounds: [2]float64{1e-3, 1e3}},
		K2: &kernels.RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}},
	}}
	gp := NewRegressor(kernel)
	X := mat.NewDense(20, 1, nil)
	//X = np.linspace(0.1, 9.9, 20)
	{
		x := X.RawMatrix().Data
		for i := range x {
			x[i] = .1 + ((9.9 - .1) * float64(i) / (float64(len(x) - 1)))
		}
	}
	Ymean := gp.Predict(X, nil)
	if mat.Norm(Ymean, math.Inf(1)) > 0 {
		t.Error("unfitted predict, expected 0")
	}
}
