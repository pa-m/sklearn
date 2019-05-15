package preprocessing

import (
	"fmt"
	"math"
	_ "math"
	"sort"
	"testing"

	"golang.org/x/exp/rand"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var _ = []Transformer{&MinMaxScaler{}, &StandardScaler{}, &RobustScaler{}, &PolynomialFeatures{}, &OneHotEncoder{}, &Shuffler{}, &Binarizer{}, &MaxAbsScaler{}, &Normalizer{}, &KernelCenterer{}, &QuantileTransformer{}}

func ExampleMinMaxScaler() {
	// adapted from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
	data := mat.NewDense(4, 2, []float64{-1., 2, -.5, 6, 0, 10, 1, 18})
	scaler := NewMinMaxScaler([]float64{0, 1})
	scaler.Fit(data, nil)
	fmt.Println(mat.Formatted(scaler.DataMax))
	X1, _ := scaler.Transform(data, nil)
	fmt.Println(mat.Formatted(X1))
	X2, _ := scaler.Transform(mat.NewDense(1, 2, []float64{2, 2}), nil)
	fmt.Println(mat.Formatted(X2))
	// Output:
	// [ 1  18]
	// ⎡   0     0⎤
	// ⎢0.25  0.25⎥
	// ⎢ 0.5   0.5⎥
	// ⎣   1     1⎦
	// [1.5    0]

}
func ExampleStandardScaler() {
	// adapted from example in http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
	data := mat.NewDense(4, 2, []float64{0, 0, 0, 0, 1, 1, 1, 1})
	scaler := NewStandardScaler()
	scaler.Fit(data, nil)
	fmt.Println(mat.Formatted(scaler.Mean))
	X1, _ := scaler.Transform(data, nil)
	fmt.Println(mat.Formatted(X1))
	X2, _ := scaler.Transform(mat.NewDense(1, 2, []float64{2, 2}), nil)
	fmt.Println(mat.Formatted(X2))
	X3, _ := scaler.InverseTransform(mat.NewDense(1, 2, []float64{3, 3}), nil)
	fmt.Println(mat.Formatted(X3))

	// Output:
	// [0.5  0.5]
	// ⎡-1  -1⎤
	// ⎢-1  -1⎥
	// ⎢ 1   1⎥
	// ⎣ 1   1⎦
	// [3  3]
	// [2  2]
}

func TestStandardScaler(t *testing.T) {

	m := NewStandardScaler()
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{8, 8, 8})
	Y, _ := m.Transform(X, nil)
	//fmt.Printf("Y=%#v\n", Y)
	X2, _ := m.InverseTransform(Y, nil)
	if !floats.EqualApprox(X.RawRowView(0), X2.RawRowView(0), 1e-6) {
		t.Errorf("StandardScaler inversetransform failed %v", X2.RawRowView(0))
		t.Fail()
	}
}

func TestRobustScaler(t *testing.T) {
	m := NewDefaultRobustScaler()
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{8, 8, 8})
	Y, _ := m.Transform(X, nil)
	X2, _ := m.InverseTransform(Y, nil)
	if !floats.EqualApprox(X.RawRowView(0), X2.RawRowView(0), 1e-6) {
		t.Errorf("RobustScaler inversetransform failed %v", X2.RawRowView(0))
		t.Fail()
	}
}

func TestRobustScalerCenter(t *testing.T) {
	m := NewRobustScaler(true, false, nil)
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 4, 7, 3, 2, 8, 9, 1, 1})
	correctY := mat.NewDense(3, 3, []float64{-2, 2, 0, 0, 0, 1, 6, -1, -6})
	Y, _ := m.FitTransform(X, nil)
	if !mat.Equal(Y, correctY) {
		t.Errorf("RobustScaler hand-crafted centering test failed - should be\n%v\nbut got:\n%v",
			mat.Formatted(correctY, mat.Prefix(""), mat.Squeeze()),
			mat.Formatted(Y, mat.Prefix(""), mat.Squeeze()))
		t.Fail()
	}
}

func TestRobustScalerQuantiles(t *testing.T) {
	m := NewRobustScaler(false, true, nil) // Use default (0.25, 0.75)
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(8, 1, []float64{9, 10, 12, 13, 19, 20, 21, 22})
	correctY := mat.NewDense(8, 1, []float64{0.9, 1.0, 1.2, 1.3, 1.9, 2.0, 2.1, 2.2})
	Y, _ := m.FitTransform(X, nil)
	if !mat.Equal(Y, correctY) {
		t.Errorf("RobustScaler hand-crafted quantiles test failed - should be\n%v\nbut got:\n%v",
			mat.Formatted(correctY, mat.Prefix(""), mat.Squeeze()),
			mat.Formatted(Y, mat.Prefix(""), mat.Squeeze()))
		t.Fail()
	}
}

func ExampleRobustScaler() {
	m := NewRobustScaler(true, false, nil)
	X := mat.NewDense(3, 3, []float64{1, 4, 7, 3, 2, 8, 9, 1, 1})
	//correctY := mat.NewDense(3, 3, []float64{-2, 2, 0, 0, 0, 1, 6, -1, -6})
	X1, _ := m.FitTransform(X, nil)
	fmt.Printf("centered:\n%g\n", mat.Formatted(X1))

	m = NewRobustScaler(false, true, nil) // Use default (0.25, 0.75)
	X = mat.NewDense(8, 1, []float64{9, 10, 12, 13, 19, 20, 21, 22})
	X1, _ = m.FitTransform(X, nil)
	fmt.Printf("quantiles:\n%g\n", mat.Formatted(X1))
	// Output:
	// centered:
	// ⎡-2   2   0⎤
	// ⎢ 0   0   1⎥
	// ⎣ 6  -1  -6⎦
	// quantiles:
	// ⎡0.9⎤
	// ⎢  1⎥
	// ⎢1.2⎥
	// ⎢1.3⎥
	// ⎢1.9⎥
	// ⎢  2⎥
	// ⎢2.1⎥
	// ⎣2.2⎦
}

func TestPolynomialFeatures(t *testing.T) {
	nSamples, nFeatures := 1, 3
	X := mat.NewDense(nSamples, nFeatures, []float{1, 2, 3})
	var expected string
	var pf *PolynomialFeatures

	pf = NewPolynomialFeatures(2)
	pf.Fit(mat.NewDense(1, 2, nil), nil)
	expected = "[[0 0] [1 0] [0 1] [2 0] [1 1] [0 2]]"
	if fmt.Sprintf("%v", pf.Powers) != expected {
		t.Errorf("expected:\n%s\ngot:\n%s", expected, fmt.Sprintf("%v", pf.Powers))

	}

	pf = NewPolynomialFeatures(3)
	isTransformer := func(Transformer) {}
	isTransformer(pf)
	pf.IncludeBias = true
	pf.InteractionOnly = false
	pf.Fit(X, nil)
	expected = "[[0 0 0] [1 0 0] [0 1 0] [0 0 1] [2 0 0] [1 1 0] [1 0 1] [0 2 0] [0 1 1] [0 0 2] [3 0 0] [2 1 0] [2 0 1] [1 2 0] [1 1 1] [1 0 2] [0 3 0] [0 2 1] [0 1 2] [0 0 3]]"
	if fmt.Sprintf("%v", pf.Powers) != expected {
		t.Errorf("expected:\n%s\ngot:\n%s", expected, fmt.Sprintf("%v", pf.Powers))

	}
	Xout, _ := pf.Transform(X, nil)
	expected = "[1 1 2 3 1 2 3 4 6 9 1 2 3 4 6 9 8 12 18 27]"
	if expected != fmt.Sprint(Xout.RawRowView(0)) {
		t.Errorf("polyfeatures transform\nexpected %v\ngot %v", expected, fmt.Sprint(Xout.RawRowView(0)))
	}
	X1, _ := pf.Transform(X, nil)
	X2, _ := pf.InverseTransform(X1, nil)
	Xd := mat.NewDense(nSamples, nFeatures, nil)
	Xd.Sub(X2, X)
	if mat.Norm(Xd, 2) > 1e-4 {
		t.Errorf("PolynomialFeatures.InverseTransform failed expected %#v, got %#v", X.RawRowView(0), X2.RawRowView(0))
	}

	pf.IncludeBias = true
	pf.InteractionOnly = true
	pf.Fit(X, nil)
	//fmt.Printf("powers interactiononly=%v\n", pf.Powers)
	expected = "[[0 0 0] [1 0 0] [0 1 0] [0 0 1] [1 1 0] [1 0 1] [0 1 1] [1 1 1]]"
	if fmt.Sprintf("%v", pf.Powers) != expected {
		t.Errorf("interactions_only expected:\n%s\ngot:\n%s", expected, fmt.Sprintf("%v", pf.Powers))

	}

	pf.IncludeBias = false
	pf.InteractionOnly = true
	pf.Fit(X, nil)
	expected = "[[1 0 0] [0 1 0] [0 0 1] [1 1 0] [1 0 1] [0 1 1] [1 1 1]]"
	//fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != expected {
		t.Errorf("polyfeatures nobias nointeraction\nexpected %v\ngot %v",
			expected,
			fmt.Sprintf("%v", pf.Powers),
		)
	}

}

func ExampleAddDummyFeature() {
	X := mat.NewDense(2, 5, []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	AddDummyFeature(X)
	fmt.Printf("X %v\n", X.RawRowView(0))
	fmt.Printf("X %v\n", X.RawRowView(1))
	// Output:
	// X [1 2 3 4 5 6]
	// X [1 7 8 9 10 11]
}

/*
		>>> from sklearn.preprocessing import OneHotEncoder
	>>> enc = OneHotEncoder()
	>>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
	OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
	       handle_unknown='error', n_values='auto', sparse=True)
	>>> enc.n_values_
	array([2, 3, 4])
	>>> enc.feature_indices_
	array([0, 2, 5, 9])
	>>> enc.transform([[0, 1, 1]]).toarray()
	array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
*/

func ExampleOneHotEncoder() {
	// adapted from example in http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
	enc := NewOneHotEncoder()
	X, Y := mat.NewDense(4, 3, []float64{0, 0, 3, 1, 1, 0, 0, 2, 1, 1, 0, 2}), (*mat.Dense)(nil)
	enc.Fit(X, Y)
	fmt.Println(enc.NValues)
	fmt.Println(enc.FeatureIndices)
	X0 := mat.NewDense(1, 3, []float64{0, 1, 1})
	X1, _ := enc.Transform(X0, nil)
	fmt.Println(mat.Formatted(X1))
	X2, _ := enc.InverseTransform(X1, nil)
	fmt.Println(mat.Formatted(X2))
	// Output:
	// [2 3 4]
	// [0 2 5 9]
	// [1  0  0  1  0  0  1  0  0]
	// [0  1  1]
}

func ExampleShuffler() {
	X, Y := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}), mat.NewDense(2, 3, []float64{7, 8, 9, 10, 11, 12})
	m := NewShuffler()
	m.RandomState = base.NewSource(7)
	X1, Y1 := m.FitTransform(X, Y)

	fmt.Println("Transformed:")
	fmt.Printf("%s", base.MatStr(X1, Y1))
	X2, Y2 := m.InverseTransform(X1, Y1)
	fmt.Println("InverseTransformed:")
	fmt.Printf("%s", base.MatStr(X2, Y2))

	// Output:
	// Transformed:
	// 4	5	6	10	11	12
	// 1	2	3	7	8	9
	// InverseTransformed:
	// 1	2	3	7	8	9
	// 4	5	6	10	11	12
}

func TestTransformer(t *testing.T) {
	f := func(t Transformer) Transformer { return t }
	f(NewStandardScaler())
	f(NewMinMaxScaler([]float64{0, 1}))
	f(NewOneHotEncoder())
	f(NewPolynomialFeatures(2))
	f(NewShuffler())
}

func ExampleMaxAbsScaler() {
	mas := NewMaxAbsScaler()
	X0 := mat.NewDense(2, 3, []float64{1, 2, 0, 3, -4, 0})
	X1, _ := mas.FitTransform(X0, nil)
	X2, _ := mas.InverseTransform(X1, nil)
	fmt.Println("MaxAbs", mas.MaxAbs)
	fmt.Println("Scale ", mas.Scale)
	fmt.Printf("Scaled:\n%g\n", mat.Formatted(X1))
	fmt.Printf("Unscaled:\n%g\n", mat.Formatted(X2))
	// Output:
	// 	MaxAbs [3 4 0]
	// Scale  [3 4 1]
	// Scaled:
	// ⎡0.3333333333333333                 0.5                   0⎤
	// ⎣                 1                  -1                   0⎦
	// Unscaled:
	// ⎡ 1   2   0⎤
	// ⎣ 3  -4   0⎦

}

func ExampleBinarizer() {
	// adapted from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
	X := mat.NewDense(3, 3, []float64{1, -1, 2, 2, 0, 0, 0, 1, -1})
	binarizer := NewBinarizer()
	binarizer.Fit(X, nil) // fit does nothing
	X1, _ := binarizer.Transform(X, nil)
	fmt.Println(mat.Formatted(X1))

	// It is possible to adjust the threshold of the binarizer:
	binarizer.Threshold = 1.1
	X1, _ = binarizer.Transform(X, nil)
	fmt.Println(mat.Formatted(X1))
	// Output:
	// ⎡1  0  1⎤
	// ⎢1  0  0⎥
	// ⎣0  1  0⎦
	// ⎡0  0  1⎤
	// ⎢1  0  0⎥
	// ⎣0  0  0⎦

}

func ExampleNormalizer() {
	// adapted from example in http://scikit-learn.org/stable/modules/preprocessing.html#normalization
	X := mat.NewDense(3, 3, []float64{
		1, -1, 2,
		2, 0, 0,
		0, 1, -1})
	Xnormalized, _ := NewNormalizer().FitTransform(X, nil)
	fmt.Printf("%.3f\n", mat.Formatted(Xnormalized))
	// Output:
	// ⎡ 0.408  -0.408   0.816⎤
	// ⎢ 1.000   0.000   0.000⎥
	// ⎣ 0.000   0.707  -0.707⎦

}

func ExampleScale() {
	// adapted from http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
	Xtrain := mat.NewDense(3, 3, []float64{1, -1, 2, 2, 0, 0, 0, 1, -1})
	Xscaled := Scale(Xtrain)
	fmt.Printf("Xscaled\n%.3f\n", mat.Formatted(Xscaled))
	mean := Mean(Xscaled)
	std := (NumpyLike{}).Std(Xscaled)
	fmt.Printf("mean:%g\nstd:%.3f\n", mat.Formatted(mean), mat.Formatted(std))
	// Output:
	// Xscaled
	// ⎡ 0.000  -1.225   1.336⎤
	// ⎢ 1.225   0.000  -0.267⎥
	// ⎣-1.225   1.225  -1.069⎦
	// mean:[0  0  0]
	// std:[1.000  1.000  1.000]

}

func ExampleKernelCenterer() {
	K := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	kc := NewKernelCenterer()
	kc.Fit(K, nil)
	K1, _ := kc.Transform(K, nil)
	fmt.Printf("KFitRows:%.3f\n", kc.KFitRows)
	fmt.Printf("KFitAll:%.3f\n", kc.KFitAll)
	fmt.Printf("Centered:\n%.3f\n", mat.Formatted(K1))
	// Output:
	// 	KFitRows:[4.000 5.000 6.000]
	// KFitAll:5.000
	// Centered:
	// ⎡0.000  0.000  0.000⎤
	// ⎢0.000  0.000  0.000⎥
	// ⎣0.000  0.000  0.000⎦

}

func ExampleQuantileTransformer() {
	// works approximately. results are not exact. need a pass in the other way. Avoid use
	type NormFloat64er interface{ NormFloat64() float64 }
	var rng rand.Source = base.NewSource(0)
	normal := func(loc, scale float64, n int) []float64 {
		data := make([]float64, n)
		for i := range data {
			data[i] = loc + scale*rng.(NormFloat64er).NormFloat64()
		}
		return data
	}
	X := mat.NewDense(25, 1, normal(.5, .25, 25))
	sort.Float64s(X.RawMatrix().Data)

	qt := NewQuantileTransformer(10, "uniform", rng)
	qt.Fit(X, nil)
	fmt.Printf("X=%.8f\n", mat.Formatted(X.T()))
	fmt.Printf("references=%.8f\n", qt.references)
	fmt.Printf("quantiles=%.8f\n", qt.Quantiles.T().(*mat.Dense).RawRowView(0))
	Xout, _ := qt.Transform(X, nil)
	fmt.Printf("transformed=%.8f\n", mat.Formatted(Xout.T()))

	tol := 1e-8
	expected := []float64{0.00000010, 0.09871873, 0.10643612, 0.11754671, 0.21017437,
		0.21945445, 0.23498666, 0.32443642, 0.33333333, 0.41360794,
		0.42339464, 0.46257841, 0.47112236, 0.49834237, 0.59986536,
		0.63390302, 0.66666667, 0.68873101, 0.69611125, 0.81280699,
		0.82160354, 0.88126439, 0.90516028, 0.99319435, 0.99999990}
	for i := range expected {
		if math.Abs(Xout.At(i, 0)-expected[i]) > tol {
			fmt.Printf("at %d expected %.8f, got %.8f\n", i, expected[i], Xout.At(i, 0))
		}
	}
	// Output:
	// X=[-0.13824745   0.25568053   0.28647607   0.31445874   0.44871043   0.46216070   0.47419529   0.53041875   0.53601089   0.57826693   0.58341858   0.60003930   0.60264963   0.61096581   0.66340465   0.69025943   0.71610905   0.73752210   0.74468450   0.86356838   0.87351977   0.94101309   0.96688950   1.06022330   1.06743866]
	// references=[0.00000000 0.11111111 0.22222222 0.33333333 0.44444444 0.55555556 0.66666667 0.77777778 0.88888889 1.00000000]
	// quantiles=[-0.13824745 0.30513118 0.46617223 0.53601089 0.59449906 0.62844542 0.71610905 0.82394042 0.94963856 1.06743866]
	// transformed=[0.00000010  0.09871873  0.10643612  0.11754671  0.21017437  0.21945445  0.23498666  0.32443642  0.33333333  0.41360794  0.42339464  0.46257841  0.47112236  0.49834237  0.59986536  0.63390302  0.66666667  0.68873101  0.69611125  0.81280699  0.82160354  0.88126439  0.90516028  0.99319435  0.99999990]
}

func TestYeoJohsonTransform(t *testing.T) {
	out := make([]float64, 3)
	inv := make([]float64, 3)
	col := []float64{1, 3, 4}
	tests := []struct {
		lmbda    float64
		expected []float64
	}{
		{-2, []float64{0.375, 0.46875, 0.48}},
		{2, []float64{1.5, 7.5, 12.}},
		{8.472135999999999, []float64{41.79741712785609, 14884.620423090699, 98576.97748593768}},
	}
	for _, test := range tests {
		yeoJohnsonTransform(out, col, test.lmbda)
		if !floats.EqualApprox(test.expected, out, 1e-8) {
			t.Errorf("expected %g, got %g", test.expected, out)
		}
		yeoJohnsonInverseTransform(inv, out, test.lmbda)
		if !floats.EqualApprox(col, inv, 1e-8) {
			t.Errorf("expected %g, got %g", col, inv)
		}
	}
	if math.Abs(1.3866817390302772-yeoJohnsonOptimize(col)) > 1e-8 {
		t.Errorf("yeoJohnsonOptimize failed")
	}
}
func ExamplePowerTransformer() {
	pt := NewPowerTransformer()
	data := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 2,
		4, 5,
	})
	Xout, _ := pt.FitTransform(data, nil)

	fmt.Printf("lambdas: %.4f\n", pt.Lambdas)
	fmt.Printf("transformed:\n%.4f\n", mat.Formatted(Xout))
	Xinv, _ := pt.InverseTransform(Xout, nil)
	fmt.Printf("inverse transformed:\n%.4f\n", mat.Formatted(Xinv))

	// Output:
	// lambdas: [1.3867 -3.1005]
	// transformed:
	// ⎡-1.3162  -0.7071⎤
	// ⎢ 0.2100  -0.7071⎥
	// ⎣ 1.1062   1.4142⎦
	// inverse transformed:
	// ⎡1.0000  2.0000⎤
	// ⎢3.0000  2.0000⎥
	// ⎣4.0000  5.0000⎦

}

func ExamplePowerTransformer_boxcox() {
	pt := NewPowerTransformer()
	pt.Method = "box-cox"
	data := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 2,
		4, 5,
	})
	Xout, _ := pt.FitTransform(data, nil)

	fmt.Printf("lambdas: %.4f\n", pt.Lambdas)
	fmt.Printf("transformed:\n%.4f\n", mat.Formatted(Xout))
	Xinv, _ := pt.InverseTransform(Xout, nil)
	fmt.Printf("inverse transformed:\n%.4f\n", mat.Formatted(Xinv))

	// Output:
	// lambdas: [1.0517 -2.3455]
	// transformed:
	// ⎡-1.3327  -0.7071⎤
	// ⎢ 0.2565  -0.7071⎥
	// ⎣ 1.0762   1.4142⎦
	// inverse transformed:
	// ⎡1.0000  2.0000⎤
	// ⎢3.0000  2.0000⎥
	// ⎣4.0000  5.0000⎦

}
