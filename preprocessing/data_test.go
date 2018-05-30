package preprocessing

import (
	"fmt"
	_ "math"
	"testing"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestMinMaxScaler(t *testing.T) {
	m := NewMinMaxScaler([]float{0, 1})
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 1, 5, 9})
	m.Fit(X, nil)
	if !floats.EqualApprox(m.Scale.RawRowView(0), []float{1, 1. / 3, 1. / 6}, 1e-6) {
		t.Error("bad scale")
		t.Fail()
	}
	X = mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{1, 2, 3})
	Y, _ := m.Transform(X, nil)
	if !floats.EqualApprox(Y.RawRowView(0), []float{0, 0, 0}, 1e-6) {
		t.Error("bad min")
		t.Fail()
	}

	X = mat.NewDense(1, 3, []float64{9, 5, 9})
	Y, _ = m.Transform(X, nil)
	if !floats.EqualApprox(Y.RawRowView(0), []float{1, 1, 1}, 1e-6) {
		t.Errorf("bad Y=%v\n", Y)
		t.Fail()
	}

	m = NewMinMaxScaler([]float{0, 10})
	X = mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{8, 8, 8})
	Y, _ = m.Transform(X, nil)
	X2, _ := m.InverseTransform(Y, nil)
	if !floats.EqualApprox(X.RawRowView(0), X2.RawRowView(0), 1e-6) {
		t.Errorf("MinMaxScaler InverseTransform failed %v", X2.RawRowView(0))
		t.Fail()
	}
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

func TestPolynomialFeatures(t *testing.T) {
	pf := NewPolynomialFeatures(3)
	isTransformer := func(Transformer) {}
	isTransformer(pf)
	pf.IncludeBias = true
	pf.InteractionOnly = false
	nSamples, nFeatures := 1, 3
	X := mat.NewDense(nSamples, nFeatures, []float{1, 2, 3})
	pf.Fit(X, nil)
	//fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Fail()
	}
	Xout, _ := pf.Transform(X, nil)
	if "[1 3 9 27 2 6 18 4 12 8 1 3 9 2 6 4 1 3 2 1]" != fmt.Sprint(Xout.RawRowView(0)) {
		t.Errorf("polyfeatures transform\nexpected %v\ngot %v", "[1 3 9 27 2 6 18 4 12 8 1 3 9 2 6 4 1 3 2 1]", fmt.Sprint(Xout.RawRowView(0)))
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
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 2 0] [0 3 0] [1 0 0] [2 0 0] [3 0 0]]" {
		fmt.Println("failed interactiononly")
		t.Fail()

	}

	pf.IncludeBias = false
	pf.InteractionOnly = false
	pf.Fit(X, nil)
	//fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Errorf("polyfeatures nobias nointeraction\nexpected %v\ngot %v",
			"[[0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]",
			fmt.Sprintf("%v", pf.Powers),
		)
	}

}

func ExampleInsertOnes() {
	X := mat.NewDense(2, 5, []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	InsertOnes(X)
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
	m.Fit(X, Y)
	copy(m.Perm, []int{1, 0})
	m.Transform(X, Y)
	fmt.Println("Transformed:")
	fmt.Println(base.MatStr(X, Y))
	m.InverseTransform(X, Y)
	fmt.Println("InverseTransformed:")
	fmt.Println(base.MatStr(X, Y))
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
