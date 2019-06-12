package preprocessing

import (
	"fmt"
	"math"
	"sort"

	"golang.org/x/exp/rand"

	"github.com/pa-m/sklearn/base"

	"github.com/pa-m/optimize"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type float = float64

// Transformer is an interface for various preprocessors
type Transformer = base.Transformer

// InverseTransformer is a transformer able to inverse his tranformation
type InverseTransformer interface {
	Transformer
	InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense)
}

// MinMaxScaler rescale data between FeatureRange
type MinMaxScaler struct {
	FeatureRange                            []float
	Scale, Min, DataMin, DataMax, DataRange *mat.Dense
	NSamplesSeen                            int
}

// NewMinMaxScaler creates an *MinMaxScaler with FeatureRange 0..1
func NewMinMaxScaler(featureRange []float) *MinMaxScaler {
	return &MinMaxScaler{FeatureRange: featureRange}
}

// TransformerClone ...
func (scaler *MinMaxScaler) TransformerClone() base.Transformer {
	var clone = *scaler
	return &clone
}

// Reset resets scaler to its initial state
func (scaler *MinMaxScaler) Reset() *MinMaxScaler {
	scaler.NSamplesSeen = 0
	return scaler
}

// Fit computes Sale and Min
func (scaler *MinMaxScaler) Fit(X, Y mat.Matrix) base.Fiter {
	scaler.Reset()
	return scaler.PartialFit(X, Y)
}

// PartialFit updates Scale and Min with partial data
func (scaler *MinMaxScaler) PartialFit(Xmatrix, Ymatrix mat.Matrix) Transformer {
	X := base.ToDense(Xmatrix)
	nSamples, nFeatures := X.Dims()
	if nSamples == 0 {
		return scaler
	}
	if scaler.NSamplesSeen == 0 {
		scaler.DataRange = mat.NewDense(1, nFeatures, nil)
		scaler.DataMin = mat.NewDense(1, nFeatures, nil)
		scaler.DataMax = mat.NewDense(1, nFeatures, nil)
		scaler.Min = mat.NewDense(1, nFeatures, nil)
		scaler.Scale = mat.NewDense(1, nFeatures, nil)

		scaler.DataMin.CloneFrom(X.RowView(0).T())
		scaler.DataMax.CloneFrom(X.RowView(0).T())
	}
	X.Apply(func(i int, j int, x float64) float64 {
		scaler.DataMin.Set(0, j, math.Min(scaler.DataMin.At(0, j), x))
		scaler.DataMax.Set(0, j, math.Max(scaler.DataMax.At(0, j), x))
		return x
	}, X)
	scaler.NSamplesSeen += nSamples
	// dataRange = dataMax - dataMin
	scaler.DataRange.Sub(scaler.DataMax, scaler.DataMin)
	// handle HandleZerosInScale
	scaler.DataRange.Apply(func(i int, j int, x float64) float64 {
		if x == 0. {
			return 1.
		}
		return x
	}, scaler.DataRange)
	//scaler.scale_ = ((featureRange[1] - featureRange[0]) /HandleZerosInScale(dataRange))
	scaler.Scale.Apply(func(i int, j int, x float64) float64 {
		return (scaler.FeatureRange[1] - scaler.FeatureRange[0]) / x
	}, scaler.DataRange)
	//scaler.min_ = featureRange[0] - dataMin * scaler.scale_
	scaler.Min.Apply(func(i int, j int, x float64) float64 {
		return scaler.FeatureRange[0] - scaler.DataMin.At(i, j)*scaler.Scale.At(i, j)
	}, scaler.DataRange)
	return scaler
}

// Transform applies scaling to X
func (scaler *MinMaxScaler) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return scaler.Min.At(0, j) + x*scaler.Scale.At(0, j)
	}, X)
	return Xout, base.ToDense(Y)
}

// FitTransform fit to dat, then transform it
func (scaler *MinMaxScaler) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	scaler.Fit(X, Y)
	return scaler.Transform(X, Y)
}

// InverseTransform rescale data into original bounds
func (scaler *MinMaxScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	if X == nil {
		return X, Y
	}
	nSamples, nFeatures := X.Dims()
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return (x - scaler.Min.At(0, j)) / scaler.Scale.At(0, j)
	}, X)
	return Xout, Y
}

// StandardScaler scales data by removing Mean and dividing by stddev
type StandardScaler struct {
	WithMean, WithStd bool
	Scale, Mean, Var  *mat.Dense
	NSamplesSeen      int
}

// NewStandardScaler creates a *StandardScaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{WithMean: true, WithStd: true}
}

// TransformerClone ...
func (scaler *StandardScaler) TransformerClone() base.Transformer {
	var clone = *scaler
	return &clone
}

// Reset ...
func (scaler *StandardScaler) Reset() *StandardScaler {
	scaler.NSamplesSeen = 0
	return scaler
}

// Fit computes Mean snd Std
func (scaler *StandardScaler) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	scaler.Reset()
	return scaler.PartialFit(X, Y)
}

// PartialFit computes Mean and Std
func (scaler *StandardScaler) PartialFit(X, Y *mat.Dense) Transformer {
	nSamples, nFeatures := X.Dims()
	if nSamples == 0 {
		return scaler
	}
	if scaler.NSamplesSeen == 0 {
		scaler.Var = mat.NewDense(1, nFeatures, nil)
		scaler.Mean = mat.NewDense(1, nFeatures, nil)
		scaler.Scale = mat.NewDense(1, nFeatures, nil)

	}
	scaler.Mean, scaler.Var, scaler.NSamplesSeen = IncrementalMeanAndVar(X, scaler.Mean, scaler.Var, scaler.NSamplesSeen)
	scaler.Scale.Apply(func(i int, j int, vj float64) float64 {
		if vj == 0. {
			vj = 1.
		}
		return math.Sqrt(vj)
	}, scaler.Var)
	return scaler
}

// Transform scales data
func (scaler *StandardScaler) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	Xmat := base.ToDense(X).RawMatrix()
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for i := 0; i < Xmat.Cols; i++ {
		mean, scale := 0., 1.
		if scaler.WithMean {
			mean = scaler.Mean.At(0, i)
		}
		if scaler.WithStd {
			scale = scaler.Scale.At(0, i)
		}
		for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {

			Xoutmat.Data[jXout+i] = (Xmat.Data[jX+i] - mean) / scale
		}
	}
	return Xout, base.ToDense(Y)
}

// FitTransform fit to dat, then transform it
func (scaler *StandardScaler) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	scaler.Fit(X, Y)
	return scaler.Transform(X, Y)
}

// InverseTransform unscales data
func (scaler *StandardScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	if X == nil {
		return X, Y
	}
	Xmat := X.RawMatrix()
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for i := 0; i < Xmat.Cols; i++ {
		mean, scale := 0., 1.
		if scaler.WithMean {
			mean = scaler.Mean.At(0, i)
		}
		if scaler.WithStd {
			scale = scaler.Scale.At(0, i)
		}
		for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {

			Xoutmat.Data[jXout+i] = mean + Xmat.Data[jX+i]*scale
		}
	}
	return Xout, Y
}

// Scale provides a quick and easy way to perform this operation on a single array-like dataset
func Scale(X *mat.Dense) *mat.Dense {
	Xout, _ := NewStandardScaler().FitTransform(X, nil)
	return Xout
}

// MeanStdDev for mat.Matrix
func MeanStdDev(X mat.Matrix) (mean, std *mat.Dense) {
	m, n := X.Dims()
	mean, std = mat.NewDense(1, n, nil), mat.NewDense(1, n, nil)
	base.Parallelize(-1, n, func(th, start, end int) {
		tmp := make([]float64, m)
		for col := start; col < end; col++ {
			mat.Col(tmp, col, X)
			meancol, stdcol := stat.MeanStdDev(tmp, nil)
			mean.Set(0, col, meancol)
			std.Set(0, col, stdcol*float64(n-1)/float64(n))
		}
	})

	return
}

// Mean for mat.Matrix
func Mean(X mat.Matrix) (mean *mat.Dense) {
	m, n := X.Dims()
	mean = mat.NewDense(1, n, nil)
	base.Parallelize(-1, n, func(th, start, end int) {
		tmp := make([]float64, m)
		for col := start; col < end; col++ {
			mat.Col(tmp, col, X)
			meancol := stat.Mean(tmp, nil)
			mean.Set(0, col, meancol)
		}
	})
	return
}

// NumpyLike is a namespace for numpy-like var and std
type NumpyLike struct{}

// Var for NumpyLike
func (NumpyLike) Var(X mat.Matrix) *mat.Dense {
	m, n := X.Dims()
	mean := Mean(X)
	variance := mat.NewDense(1, n, nil)
	base.Parallelize(-1, n, func(th, start, end int) {
		tmp := make([]float64, m)
		for col := start; col < end; col++ {
			mat.Col(tmp, col, X)
			floats.AddConst(-mean.At(0, col), tmp)
			floats.Mul(tmp, tmp)
			variance.Set(0, col, floats.Sum(tmp)/float64(n))
		}
	})
	return variance
}

// Std for NumpyLike
func (m NumpyLike) Std(X mat.Matrix) *mat.Dense {
	tmp := m.Var(X)
	rm := tmp.RawMatrix()
	for i := 0; i < rm.Cols; i++ {
		rm.Data[i] = math.Sqrt(rm.Data[i])
	}
	return tmp
}

//======================================================================

// RobustScaler scales data by removing centering around the Median and
// removing outliers by Quantile. See python sklearn's RobustScaler
// http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html.
//
type RobustScaler struct {
	Center          bool
	Scale           bool
	Quantiles       *QuantilePair
	Median          *mat.Dense
	Tmp             *mat.Dense
	QuantileDivider *mat.Dense
}

// QuantilePair represents bounds of quantile
type QuantilePair struct {
	Left  float64
	Right float64
}

// NewDefaultRobustScaler supplies typical arguments (via python sklearn)
func NewDefaultRobustScaler() *RobustScaler {
	return NewRobustScaler(true, true, &QuantilePair{0.25, 0.75})
}

// TransformerClone ...
func (scaler *RobustScaler) TransformerClone() base.Transformer {
	var clone = *scaler
	return &clone
}

// NewRobustScaler creates a *RobustScaler
func NewRobustScaler(center bool, scale bool, quantiles *QuantilePair) *RobustScaler {
	if scale && (quantiles == nil) {
		quantiles = &QuantilePair{0.25, 0.75}
	}
	return &RobustScaler{
		Center:    center,
		Scale:     scale,
		Quantiles: quantiles,
	}
}

// Reset ...
func (scaler *RobustScaler) Reset() *RobustScaler {
	return scaler
}

// Fit computes Median and Quantiles
func (scaler *RobustScaler) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	scaler.Reset()
	return scaler.PartialFit(Xmatrix, Ymatrix)
}

// PartialFit computes Median and Quantiles
func (scaler *RobustScaler) PartialFit(Xmatrix, Ymatrix mat.Matrix) Transformer {
	X := base.ToDense(Xmatrix)
	nSamples, nFeatures := X.Dims()
	if nSamples == 0 {
		return scaler
	}
	if scaler.Center && (scaler.Median == nil) {
		scaler.Median = mat.NewDense(1, nFeatures, nil)
	}
	if scaler.Scale && (scaler.QuantileDivider == nil) {
		scaler.QuantileDivider = mat.NewDense(1, nFeatures, nil)
	}
	if scaler.Tmp == nil {
		scaler.Tmp = mat.NewDense(1, nSamples, nil)
	}

	for c := 0; c < nFeatures; c++ {
		for r := 0; r < nSamples; r++ {
			scaler.Tmp.Set(0, r, X.At(r, c))
		}
		sort.Float64s(scaler.Tmp.RawRowView(0))
		if scaler.Center {
			scaler.Median.Set(0, c, stat.Quantile(0.5, stat.Empirical, scaler.Tmp.RawRowView(0), nil))
		}
		if scaler.Scale {
			q1 := stat.Quantile(scaler.Quantiles.Left, stat.Empirical, scaler.Tmp.RawRowView(0), nil)
			q2 := stat.Quantile(scaler.Quantiles.Right, stat.Empirical, scaler.Tmp.RawRowView(0), nil)
			scaler.QuantileDivider.Set(0, c, q2-q1)
			scaler.QuantileDivider.Apply(
				func(r int, c int, v float64) float64 {
					if v == 0.0 {
						return 1.0
					}
					return v

				},
				scaler.QuantileDivider)
		}
	}
	return scaler
}

// Transform scales data
func (scaler *RobustScaler) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	Xout = mat.DenseCopyOf(X)
	Xout.Apply(func(i int, j int, x float64) float64 {
		res := x
		if scaler.Center {
			res = res - scaler.Median.At(0, j)
		}
		if scaler.Scale {
			res = res / scaler.QuantileDivider.At(0, j)
		}
		return res
	}, X)
	return Xout, base.ToDense(Y)
}

// FitTransform fit to dat, then transform it
func (scaler *RobustScaler) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	scaler.Fit(X, Y)
	return scaler.Transform(X, Y)
}

// InverseTransform unscales data
func (scaler *RobustScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	if X == nil {
		return X, Y
	}
	Xout = mat.DenseCopyOf(X)
	Xout.Apply(func(i int, j int, x float64) float64 {
		res := x
		if scaler.Scale {
			res = res * scaler.QuantileDivider.At(0, j)
		}
		if scaler.Center {
			res = res + scaler.Median.At(0, j)
		}
		return res
	}, X)
	return Xout, Y
}

//======================================================================

// IncrementalMeanAndVar Calculate mean update and a Youngs and Cramer variance update.
// lastMean and lastVariance are statistics computed at the last step by the
// function. Both must be initialized to 0.0. In case no scaling is required
// lastVariance can be None. The mean is always required and returned because
// necessary for the calculation of the variance. lastNSamplesSeen is the
// number of samples encountered until now.
// From the paper "Algorithms for computing the sample variance: analysis and
// recommendations", by Chan, Golub, and LeVeque.
// Parameters
// ----------
// X : array-like, shape (nSamples, nFeatures)
//     Data to use for variance update
// lastMean : array-like, shape: (nFeatures,)
// lastVariance : array-like, shape: (nFeatures,)
// lastSampleCount : int
// Returns
// -------
// updatedMean : array, shape (nFeatures,)
// updatedVariance : array, shape (nFeatures,)
//     If None, only mean is computed
// updatedSampleCount : int
// References
// ----------
// T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
//     variance: recommendations, The American Statistician, Vol. 37, No. 3,
//     pp. 242-247
// Also, see the sparse implementation of this in
// `utils.sparsefuncs.incrMeanVarianceAxis` and
// `utils.sparsefuncsFast.incrMeanVarianceAxis0`
// """
func IncrementalMeanAndVar(X, lastMean, lastVariance *mat.Dense,
	lastSampleCount int) (updatedMean, updatedVariance *mat.Dense, updatedSampleCount int) {
	// # old = stats until now
	// # new = the current increment
	// # updated = the aggregated stats
	//lastSum := lastMean * lastSampleCount
	newSampleCount, nFeatures := X.Dims()
	lastSum := mat.DenseCopyOf(lastMean)
	lastSum.Scale(float(lastSampleCount), lastSum)
	// new sum
	newSum := mat.NewDense(1, nFeatures, nil)
	for i := 0; i < newSampleCount; i++ {
		newSum.Add(newSum, X.RowView(i).T())
	}

	updatedSampleCount = lastSampleCount + newSampleCount

	//updatedMean = (lastSum + newSum) / updatedSampleCount
	updatedMean = mat.NewDense(1, nFeatures, nil)
	updatedMean.Add(lastSum, newSum)
	updatedMean.Scale(1./float(updatedSampleCount), updatedMean)

	//newUnnormalizedVariance = X.var(axis=0) * newSampleCount
	newUnnormalizedVariance := mat.NewDense(1, nFeatures, nil)
	updatedUnnormalizedVariance := mat.NewDense(1, nFeatures, nil)
	newMean := mat.NewDense(1, nFeatures, nil)
	newMean.Scale(1./float(newSampleCount), newSum)

	tmp := mat.NewDense(1, nFeatures, nil)
	for i := 0; i < newSampleCount; i++ {
		tmp.Sub(X.RowView(i).T(), newMean)
		tmp.MulElem(tmp, tmp)
		newUnnormalizedVariance.Add(newUnnormalizedVariance, tmp)
	}

	if lastSampleCount == 0 { //# Avoid division by 0
		updatedUnnormalizedVariance.CloneFrom(newUnnormalizedVariance)
	} else {
		lastOverNewCount := float(lastSampleCount) / float(newSampleCount)
		//lastUnnormalizedVariance := lastVariance * lastSampleCount
		lastUnnormalizedVariance := mat.NewDense(1, nFeatures, nil)
		lastUnnormalizedVariance.CloneFrom(lastVariance)
		lastUnnormalizedVariance.Scale(float(lastSampleCount), lastUnnormalizedVariance)

		// updatedUnnormalizedVariance = (
		//     lastUnnormalizedVariance +
		//     newUnnormalizedVariance +
		//     lastOverNewCount / updatedSampleCount *
		//     (lastSum / lastOverNewCount - newSum) ** 2)
		tmp.CloneFrom(lastSum)
		tmp.Scale(1./lastOverNewCount, tmp)
		tmp.Add(tmp, newSum)
		tmp.Mul(tmp, tmp)
		tmp.Scale(lastOverNewCount/float(updatedSampleCount), tmp)

		updatedUnnormalizedVariance.CloneFrom(lastUnnormalizedVariance)
		updatedUnnormalizedVariance.Add(updatedUnnormalizedVariance, newUnnormalizedVariance)
		updatedUnnormalizedVariance.Add(updatedUnnormalizedVariance, tmp)
	}
	//updatedVariance = updatedUnnormalizedVariance / updatedSampleCount
	updatedVariance = mat.NewDense(1, nFeatures, nil)
	updatedVariance.CloneFrom(updatedUnnormalizedVariance)
	updatedVariance.Scale(1./float(updatedSampleCount), updatedVariance)

	return updatedMean, updatedVariance, updatedSampleCount
}

// PolynomialFeatures struct
type PolynomialFeatures struct {
	Degree                       int
	InteractionOnly, IncludeBias bool
	Powers                       [][]int
}

// NewPolynomialFeatures creates a *PolynomialFeatures
func NewPolynomialFeatures(degree int) *PolynomialFeatures {
	return &PolynomialFeatures{Degree: degree, InteractionOnly: false, IncludeBias: true}

}

// TransformerClone ...
func (poly *PolynomialFeatures) TransformerClone() base.Transformer {
	var clone = *poly
	return &clone
}

// Fit precompute Powers
// Powers[i, j] is the exponent of the jth input in the ith output.
func (poly *PolynomialFeatures) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	_, nFeatures := X.Dims()
	poly.Powers = make([][]int, 0)
	comb := combinationsWithReplacement
	if poly.InteractionOnly {
		comb = combinations
	}
	start := 0
	if !poly.IncludeBias {
		start = 1
	}
	for i := start; i <= poly.Degree; i++ {
		for c := range comb(intrange(nFeatures), i) {
			poly.Powers = append(poly.Powers, bincount(c, nFeatures))
		}
	}
	return poly
}

// Transform returns data with polynomial features added
func (poly *PolynomialFeatures) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	Xout = mat.NewDense(nSamples, len(poly.Powers), nil)
	xi, xo := base.ToDense(X).RawMatrix(), Xout.RawMatrix()
	for i, ipos, opos := 0, 0, 0; i < nSamples; i, ipos, opos = i+1, ipos+xi.Stride, opos+xo.Stride {
		for ioutput, p := range poly.Powers {
			v := 1.
			for j, pj := range p {
				v *= math.Pow(xi.Data[ipos+j], float(pj))
			}
			xo.Data[opos+ioutput] = v
		}
	}
	return Xout, base.ToDense(Y)
}

// FitTransform fit to dat, then transform it
func (poly *PolynomialFeatures) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	poly.Fit(X, Y)
	return poly.Transform(X, Y)
}

// InverseTransform inverse tranformation for PolynomialFeatures.
func (poly *PolynomialFeatures) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	if X == nil {
		return X, Y
	}
	type jt struct{ jorig, jpoly int }
	var jts []jt
	nSamples, _ := X.Dims()
	XoutCols := 0

	intsum := func(a []int) int {
		var s int
		for _, v := range a {
			s += v
		}
		return s
	}
	intmaxidx := func(a []int) int {
		i := 0
		for j, v := range a {
			if v > a[i] {
				i = j
			}
		}
		return i
	}
	for ioutput, p := range poly.Powers {
		var jMax = intmaxidx(p)
		var sumpj = intsum(p)
		if sumpj == 1 {
			//fmt.Println(ioutput, "p", p, "sumpj", sumpj, "jWith1", jMax)
			jts = append(jts, jt{jMax, ioutput})
			if jMax >= XoutCols {
				XoutCols = jMax + 1
			}
		}
	}
	Xout = mat.NewDense(nSamples, XoutCols, nil)

	for _, pair := range jts {
		//fmt.Println(jts)
		for i := 0; i < nSamples; i++ {
			Xout.Set(i, pair.jorig, X.At(i, pair.jpoly))
		}
	}
	return
}

// DenseMean puts in Xmean[1,nFeatures] the mean of X rows
func DenseMean(Xmean *mat.Dense, X mat.Matrix) *mat.Dense {
	nSamples, nFeatures := X.Dims()
	if Xmean == nil {
		Xmean = mat.NewDense(1, nFeatures, nil)
	}
	Xmean.Apply(func(i int, j int, v float64) float64 {
		for i := 0; i < nSamples; i++ {
			v += X.At(i, j)
		}
		return v / float64(nSamples)
	}, Xmean)
	return Xmean
}

// AddDummyFeature insert a column of ones to fit intercept
func AddDummyFeature(X *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	X1 := mat.NewDense(nSamples, nFeatures+1, nil)
	X1.Apply(func(i, j int, _ float64) float64 {
		if j == 0 {
			return 1
		}
		return X.At(i, j-1)
	}, X1)
	X.CloneFrom(X1)
}

// OneHotEncoder Encode categorical integer features using a one-hot aka one-of-K scheme.
type OneHotEncoder struct {
	NValues, FeatureIndices []int
	Values                  [][]float64
}

// NewOneHotEncoder creates a *OneHotEncoder
func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{}
}

// TransformerClone ...
func (m *OneHotEncoder) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit ...
func (m *OneHotEncoder) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	NSamples, NFeatures := X.Dims()
	m.NValues = make([]int, NFeatures)
	m.FeatureIndices = make([]int, NFeatures+1)
	m.Values = make([][]float64, NFeatures)
	for feature := 0; feature < NFeatures; feature++ {
		cmap := make(map[float64]bool)
		for sample := 0; sample < NSamples; sample++ {
			val := X.At(sample, feature)
			cmap[val] = true
		}
		m.NValues[feature] = len(cmap)
		m.FeatureIndices[feature+1] = m.FeatureIndices[feature] + len(cmap)
		vals := make([]float64, len(cmap))
		ival := 0
		for v := range cmap {
			vals[ival] = v
			ival++
		}
		sort.Float64s(vals)
		m.Values[feature] = vals
	}
	return m
}

// Transform transform Y labels to one hot encoded format
func (m *OneHotEncoder) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	NSamples, nfeatures := X.Dims()
	Yout = base.ToDense(Y)
	columns := 0
	for feature := 0; feature < nfeatures; feature++ {
		ovo := false
		if ovo {
			columns++
		} else {
			columns += m.NValues[feature]
		}
	}
	Xout = mat.NewDense(NSamples, columns, nil)
	baseColumn := 0
	for feature := 0; feature < nfeatures; feature++ {
		ovo := false
		if ovo {
			for sample := 0; sample < NSamples; sample++ {
				v := Y.At(sample, feature)
				Yout.Set(sample, baseColumn, v)
			}
			baseColumn++
			break
		}
		cmap := make(map[float64]int)
		for i, v := range m.Values[feature] {
			cmap[v] = i
		}
		for sample := 0; sample < NSamples; sample++ {
			v := X.At(sample, feature)
			//fmt.Printf("sample %d feature %d baseColumn %d nClasses %d y %g min:%d yint %d\n", sample, feature, baseColumn, m.NumClasses[feature], Y.At(sample, feature), m.Min[feature], yint)
			Xout.Set(sample, baseColumn+cmap[v], 1.)
		}
		baseColumn += m.NValues[feature]
	}
	return
}

// FitTransform fit to dat, then transform it
func (m *OneHotEncoder) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform compute Yout classes from one hot encoded format
func (m *OneHotEncoder) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	nFeatures := len(m.NValues)
	Yout = Y
	Xout = mat.NewDense(nSamples, nFeatures, nil)

	for feature := 0; feature < nFeatures; feature++ {
		cstart, cend := m.FeatureIndices[feature], m.FeatureIndices[feature+1]
		for sample := 0; sample < nSamples; sample++ {
			classNo := floats.MaxIdx(X.RawRowView(sample)[cstart:cend])
			Xout.Set(sample, feature, float64(classNo))
		}
	}
	return

}

// Shuffler shuffles rows of X and Y
type Shuffler struct {
	Perm        []int
	RandomState base.Source
}

// NewShuffler returns a *Shuffler
func NewShuffler() *Shuffler { return &Shuffler{} }

// TransformerClone ...
func (m *Shuffler) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit for Shuffler
func (m *Shuffler) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	Perm := rand.Perm
	if m.RandomState != base.Source(nil) {
		Perm = rand.New(m.RandomState).Perm
	}
	m.Perm = Perm(X.RawMatrix().Rows)
	return m
}

// Transform for Shuffler
func (m *Shuffler) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	Xout, Yout = mat.DenseCopyOf(X), mat.DenseCopyOf(Y)
	xmat, ymat := Xout.RawMatrix(), Yout.RawMatrix()
	var dstxpos, dstypos, j int
	var v float64
	for i, srcxpos, srcypos := 0, 0, 0; i < len(m.Perm); i, srcxpos, srcypos = i+1, srcxpos+xmat.Stride, srcypos+ymat.Stride {
		if m.Perm[i] > i {
			dstxpos, dstypos = m.Perm[i]*xmat.Stride, m.Perm[i]*ymat.Stride
			for j = 0; j < xmat.Cols; j++ {
				v = xmat.Data[srcxpos+j]
				xmat.Data[srcxpos+j] = xmat.Data[dstxpos+j]
				xmat.Data[dstxpos+j] = v
			}
			for j = 0; j < ymat.Cols; j++ {
				v = ymat.Data[srcypos+j]
				ymat.Data[srcypos+j] = ymat.Data[dstypos+j]
				ymat.Data[dstypos+j] = v
			}
		}
	}
	return
}

// FitTransform fit to dat, then transform it
func (m *Shuffler) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform for Shuffler
func (m *Shuffler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	xmat, ymat := X.RawMatrix(), Y.RawMatrix()
	var srcxpos, srcypos, dstxpos, dstypos, j int
	var v float64
	for i := len(m.Perm) - 1; i >= 0; i-- {
		if m.Perm[i] > i {
			srcxpos, dstxpos = m.Perm[i]*xmat.Stride, i*xmat.Stride
			srcypos, dstypos = m.Perm[i]*ymat.Stride, i*ymat.Stride
			for j = 0; j < xmat.Stride; j++ {
				v = xmat.Data[srcxpos+j]
				xmat.Data[srcxpos+j] = xmat.Data[dstxpos+j]
				xmat.Data[dstxpos+j] = v
			}
			for j = 0; j < ymat.Stride; j++ {
				v := ymat.Data[srcypos+j]
				ymat.Data[srcypos+j] = ymat.Data[dstypos+j]
				ymat.Data[dstypos+j] = v
			}

		}
	}

	return X, Y
}

// Binarizer Binarize data (set feature values to 0 or 1) according to a threshold
type Binarizer struct{ Threshold float64 }

// NewBinarizer ...
func NewBinarizer() *Binarizer { return &Binarizer{} }

// TransformerClone ...
func (m *Binarizer) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit for binarizer does nothing
func (m *Binarizer) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	return m
}

// Transform for Binarizer
func (m *Binarizer) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	rmx := base.ToDense(X).RawMatrix()
	Xout = mat.NewDense(rmx.Rows, rmx.Cols, nil)
	rmXout := Xout.RawMatrix()
	Yout = base.ToDense(Y)
	for r, xpos := 0, 0; r < rmx.Rows; r, xpos = r+1, xpos+rmx.Stride {
		for c := 0; c < rmx.Cols; c++ {

			if rmx.Data[xpos+c] > m.Threshold {
				rmXout.Data[xpos+c] = 1
			} else {
				rmXout.Data[xpos+c] = 0
			}
		}
	}
	return
}

// FitTransform fit to data, then transform it
func (m *Binarizer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// MaxAbsScaler ...
type MaxAbsScaler struct {
	Scale, MaxAbs []float64
	NSamplesSeen  int
}

// NewMaxAbsScaler ...
func NewMaxAbsScaler() *MaxAbsScaler { return &MaxAbsScaler{} }

// TransformerClone ...
func (m *MaxAbsScaler) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit for MaxAbsScaler ...
func (m *MaxAbsScaler) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	Xmat := X.RawMatrix()
	m.MaxAbs = make([]float64, Xmat.Cols)
	m.Scale = make([]float64, Xmat.Cols)
	return m.PartialFit(X, Y)
}

// PartialFit for MaxAbsScaler ...
func (m *MaxAbsScaler) PartialFit(X, Y *mat.Dense) base.Transformer {
	Xmat := X.RawMatrix()
	for jX := 0; jX < Xmat.Rows*Xmat.Stride; jX = jX + Xmat.Stride {
		for i, v := range Xmat.Data[jX : jX+Xmat.Cols] {
			if v < 0. {
				v = -v
			}
			if v > m.MaxAbs[i] {
				m.MaxAbs[i] = v
			}
		}
	}
	for i, v := range m.MaxAbs {
		if v > 0. {
			m.Scale[i] = v
		} else {
			m.Scale[i] = 1.
		}
	}
	m.NSamplesSeen += Xmat.Rows
	return m
}

// Transform for MaxAbsScaler ...
func (m *MaxAbsScaler) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	Xmat := base.ToDense(X).RawMatrix()
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {
		for i, v := range Xmat.Data[jX : jX+Xmat.Cols] {
			Xoutmat.Data[jXout+i] = v / m.Scale[i]
		}
	}
	Yout = base.ToDense(Y)
	return
}

// FitTransform fit to dat, then transform it
func (m *MaxAbsScaler) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform for MaxAbsScaler ...
func (m *MaxAbsScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xmat := X.RawMatrix()
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {
		for i, v := range Xmat.Data[jX : jX+Xmat.Cols] {
			Xoutmat.Data[jXout+i] = v * m.Scale[i]
		}
	}
	Yout = Y
	return
}

// Normalizer Normalize samples individually to unit norm.
// Norm is l1|l2|max. l2 by default
type Normalizer struct {
	Norm      string
	Axis      int
	nrmValues []float64
}

// NewNormalizer returns a normaliser with Norm l2 and axis 1
func NewNormalizer() *Normalizer { return &Normalizer{Norm: "l2", Axis: 1} }

// TransformerClone ...
func (m *Normalizer) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit for Normalizer ...
func (m *Normalizer) Fit(X, Y mat.Matrix) base.Fiter { return m }

// Transform for Normalizer ...
func (m *Normalizer) Transform(Xmatrix, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	X := base.ToDense(Xmatrix)
	NSamples, NFeatures := X.Dims()

	Xout = mat.NewDense(NSamples, NFeatures, nil)
	var norm = 2.
	switch m.Norm {
	case "l1":
		norm = 1.
	case "max":
		norm = math.Inf(1)
	}
	if m.Axis == 0 {
		tmp := make([]float64, NSamples)
		m.nrmValues = make([]float64, NFeatures)
		for i := 0; i < NFeatures; i++ {
			mat.Col(tmp, i, X)
			nrm := mat.Norm(mat.NewVecDense(NSamples, tmp), norm)
			m.nrmValues[i] = nrm
			if nrm != 0 {
				floats.Scale(1/nrm, tmp)
			}
			Xout.SetCol(i, tmp)
		}
	} else {
		tmp := make([]float64, NFeatures)
		m.nrmValues = make([]float64, NSamples)
		for i := 0; i < NSamples; i++ {
			mat.Row(tmp, i, X)
			nrm := mat.Norm(mat.NewVecDense(NSamples, tmp), norm)
			m.nrmValues[i] = nrm
			if nrm != 0 {
				floats.Scale(1/nrm, tmp)
			}
			Xout.SetRow(i, tmp)
		}
	}
	Yout = base.ToDense(Y)
	return
}

// FitTransform fit to dat, then transform it
func (m *Normalizer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform for Normalizer ...
func (m *Normalizer) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	if m.Axis == 0 {
		tmp := make([]float64, NSamples)
		for i := 0; i < NFeatures; i++ {
			mat.Col(tmp, i, X)
			nrm := m.nrmValues[i]
			floats.Scale(nrm, tmp)
			Xout.SetCol(i, tmp)
		}
	} else {
		tmp := make([]float64, NFeatures)
		for i := 0; i < NSamples; i++ {
			mat.Row(tmp, i, X)
			nrm := m.nrmValues[i]
			floats.Scale(nrm, tmp)
			Xout.SetRow(i, tmp)
		}
	}
	return
}

// KernelCenterer Center a kernel matrix
type KernelCenterer struct {
	KFitAll  float64
	KFitRows []float64
}

// NewKernelCenterer ...
func NewKernelCenterer() *KernelCenterer { return &KernelCenterer{} }

// TransformerClone ...
func (m *KernelCenterer) TransformerClone() base.Transformer {
	var clone = *m
	return &clone
}

// Fit for KernelCenterer ...
func (m *KernelCenterer) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X := base.ToDense(Xmatrix)
	r, c := X.Dims()
	m.KFitRows = make([]float64, c)

	for col := 0; col < c; col++ {
		Xmat := X.RawMatrix()
		s := 0.
		for jX := 0; jX < Xmat.Rows*Xmat.Stride; jX = jX + Xmat.Stride {
			s += Xmat.Data[jX+col]
		}
		m.KFitRows[col] = s / float64(r)
	}

	m.KFitAll = floats.Sum(m.KFitRows) / float64(r)
	return m
}

// Transform for KernelCenterer ...
func (m *KernelCenterer) Transform(Xmatrix, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	X := base.ToDense(Xmatrix)
	r, _ := X.Dims()
	KPredCols := make([]float64, r)
	base.Parallelize(-1, r, func(th, start, end int) {
		for i := start; i < end; i++ {
			KPredCols[i] = floats.Sum(X.RawRowView(i)) / float64(len(m.KFitRows))
		}
	})
	Xmat := X.RawMatrix()
	Xout = mat.NewDense(Xmat.Cols, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for j, jX, jXout := 0, 0, 0; j < Xmat.Rows; j, jX, jXout = j+1, jX+Xmat.Stride, jXout+Xoutmat.Stride {
		for i, v := range Xmat.Data[jX : jX+Xmat.Cols] {
			Xoutmat.Data[jXout+i] = v - m.KFitRows[i] - KPredCols[j] + m.KFitAll
		}
	}
	Yout = base.ToDense(Y)
	return
}

// FitTransform fit to dat, then transform it
func (m *KernelCenterer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// QuantileTransformer Transform features using quantiles information.
type QuantileTransformer struct {
	NQuantiles         int
	Subsample          int
	OutputDistribution string
	RandomState        rand.Source
	references         []float64
	Quantiles          mat.Matrix
}

// NewQuantileTransformer returns a new QuantileTransformer
func NewQuantileTransformer(NQuantiles int, outputDistribution string, RandomState rand.Source) *QuantileTransformer {
	return &QuantileTransformer{NQuantiles: NQuantiles, Subsample: 1e5, OutputDistribution: outputDistribution, RandomState: RandomState}
}

// Fit for QuantileTransformer retain X or a part of it
func (m *QuantileTransformer) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	nSamples, nFeatures := Xmatrix.Dims()
	XT := mat.DenseCopyOf(Xmatrix.T())
	m.references = make([]float64, m.NQuantiles)
	Q := mat.NewDense(nFeatures, m.NQuantiles, nil)
	for i := range m.references {
		m.references[i] = float64(i) / float64(m.NQuantiles-1)
	}
	for feature := 0; feature < nFeatures; feature++ {
		values := XT.RawRowView(feature)
		quantiles := Q.RawRowView(feature)
		for i := range m.references {
			realSample := float64(i) / float64(m.NQuantiles-1) * float64(nSamples-1)
			i0 := int(math.Min(math.Floor(realSample), float64(nSamples-1)))
			if realSample == float(i0) || i0+1 >= len(values) {
				quantiles[i] = values[i0]
			} else {
				p := realSample - float64(i0)
				quantiles[i] = (values[i0]*(1-p) + values[i0+1]*(p))
			}
		}
	}
	m.Quantiles = Q.T()
	return m
}

// Transform for QuantileTransformer returns Quantiles of X in Xout
func (m *QuantileTransformer) Transform(Xmatrix, Ymatrix mat.Matrix) (Xout, Yout *mat.Dense) {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	nSamples, nFeatures := X.Dims()
	eps := 1e-7
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	for c := 0; c < nFeatures && c < len(m.references); c++ {
		for q := m.NQuantiles - 2; q >= 0; q-- {
			x0, x1 := m.Quantiles.At(q, c), m.Quantiles.At(q+1, c)
			for i := 0; i < nSamples; i++ {
				x := X.At(i, c)
				if x >= x0 && x <= x1 {
					Xout.Set(i, c, math.Min(m.references[m.NQuantiles-1]-eps, math.Max(m.references[0]+eps, m.references[q]+(x-x0)/(x1-x0)*(m.references[q+1]-m.references[q]))))
				}
			}
		}
	}
	return Xout, Y
}

// FitTransform fit to data, then transforms it
func (m *QuantileTransformer) FitTransform(Xmatrix, Ymatrix mat.Matrix) (Xout, Yout *mat.Dense) {
	m.Fit(Xmatrix, Ymatrix)
	return m.Transform(Xmatrix, Ymatrix)
}

// TransformerClone ...
func (m *QuantileTransformer) TransformerClone() Transformer {
	clone := *m
	return &clone
}

// PowerTransformer apply a power transform featurewise to make data more Gaussian-like
// TODO support boxcox
type PowerTransformer struct {
	Method      string
	Standardize bool

	Lambdas []float64
	Scaler  *StandardScaler
}

// NewPowerTransformer returns a PowerTransformer with method yeo-johnson and standardize=true
func NewPowerTransformer() *PowerTransformer {
	return &PowerTransformer{Method: "yeo-johnson", Standardize: true}
}

// TransformerClone allow duplication
func (m *PowerTransformer) TransformerClone() base.Transformer {
	clone := *m
	return &clone
}

// Fit Estimate the optimal parameter lambda for each feature. The optimal lambda parameter for minimizing skewness is estimated on each feature independently using maximum likelihood.
func (m *PowerTransformer) Fit(X, Y mat.Matrix) base.Fiter {
	m.fit(X, Y, false)
	return m
}

// Fit Estimate the optimal parameter lambda for each feature. The optimal lambda parameter for minimizing skewness is estimated on each feature independently using maximum likelihood.
func (m *PowerTransformer) fit(X, Y mat.Matrix, forceTransform bool) (Xout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	m.Lambdas = make([]float64, nFeatures)
	var optimFunc func(col []float64) float64
	var transformFunc func(out, x []float64, lmbda float64)
	switch m.Method {
	case "yeo-johnson":
		optimFunc = yeoJohnsonOptimize
		transformFunc = yeoJohnsonTransform
	case "box-cox":
		optimFunc = boxCoxOptimize
		transformFunc = boxCoxTransform
	default:
		panic(fmt.Errorf("'method' must be one of ('box-cox', 'yeo-johnson'), got %s instead", m.Method))
	}
	if m.Standardize || forceTransform {
		Xout = mat.NewDense(nSamples, nFeatures, nil)
	}
	base.Parallelize(-1, nFeatures, func(th, start, end int) {
		col := make([]float64, nSamples)
		out := make([]float64, nSamples)
		for c := start; c < end; c++ {
			mat.Col(col, c, X)
			m.Lambdas[c] = optimFunc(col)
			if m.Standardize || forceTransform {
				transformFunc(out, col, m.Lambdas[c])
				Xout.SetCol(c, out)
			}
		}
	})
	if m.Standardize {
		m.Scaler = NewStandardScaler()
		if forceTransform {
			Xout, _ = m.Scaler.FitTransform(Xout, nil)
		} else {
			m.Scaler.Fit(Xout, nil)
		}
	}
	return
}

// Transform apply the power transform to each feature using the fitted lambdas
func (m *PowerTransformer) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	Xout = base.ToDense(X)
	Yout = base.ToDense(Y)
	var transformFunc func(out, x []float64, lmbda float64)
	switch m.Method {
	case "yeo-johnson":
		transformFunc = yeoJohnsonTransform
	case "box-cox":
		transformFunc = boxCoxTransform
	default:
		panic(fmt.Errorf("'method' must be one of ('box-cox', 'yeo-johnson'), got %s instead", m.Method))
	}
	base.Parallelize(-1, nFeatures, func(th, start, end int) {
		col := make([]float64, nSamples)
		out := make([]float64, nSamples)
		for c := start; c < end; c++ {
			mat.Col(col, c, X)
			transformFunc(out, col, m.Lambdas[c])
			Xout.SetCol(c, out)
		}
	})
	if m.Standardize {
		Xout, _ = m.Scaler.Transform(Xout, nil)
	}
	return
}

// FitTransform fits the data then transforms it
func (m *PowerTransformer) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	Xout = m.fit(X, Y, true)
	Yout = base.ToDense(Y)
	return
}

// InverseTransform apply the inverse power transformation using the fitted lambdas.
// The inverse of the Box-Cox transformation is given by::
// 	if lambda == 0:
// 		X = exp(X_trans)
// 	else:
// 		X = (X_trans * lambda + 1) ** (1 / lambda)
// The inverse of the Yeo-Johnson transformation is given by::
// 	if X >= 0 and lambda == 0:
// 		X = exp(X_trans) - 1
// 	elif X >= 0 and lambda != 0:
// 		X = (X_trans * lambda + 1) ** (1 / lambda) - 1
// 	elif X < 0 and lambda != 2:
// 		X = 1 - (-(2 - lambda) * X_trans + 1) ** (1 / (2 - lambda))
// 	elif X < 0 and lambda == 2:
// 		X = 1 - exp(-X_trans)
func (m *PowerTransformer) InverseTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	Yout = base.ToDense(Y)

	var inverseTransformFunc func(out, x []float64, lmbda float64)
	switch m.Method {
	case "yeo-johnson":
		inverseTransformFunc = yeoJohnsonInverseTransform
	case "box-cox":
		inverseTransformFunc = boxCoxInverseTransform
	default:
		panic(fmt.Errorf("'method' must be one of ('box-cox', 'yeo-johnson'), got %s instead", m.Method))
	}
	X1 := base.ToDense(X)
	if m.Standardize {
		X1, _ = m.Scaler.InverseTransform(X1, nil)
	}
	base.Parallelize(-1, nFeatures, func(th, start, end int) {
		col := make([]float64, nSamples)
		out := make([]float64, nSamples)
		for c := start; c < end; c++ {
			mat.Col(col, c, X1)
			inverseTransformFunc(out, col, m.Lambdas[c])
			Xout.SetCol(c, out)
		}
	})
	return
}

// yeoJohnsonOptimize Find and return optimal lambda parameter of the Yeo-Johnson
// transform by MLE, for observed data x.
// Like for Box-Cox, MLE is done via the brent optimizer.
func yeoJohnsonOptimize(x []float64) float64 {
	xTrans := make([]float64, len(x))
	negLogLikelihood := func(lmbda float64) float64 { // Return the negative log likelihood of the observed data x as a function of lambda
		nSamples := len(x)
		var sumlog1p, tmp float64
		for _, xi := range x {

			if xi > 0 {
				tmp = math.Log1p(xi)
			} else {
				tmp = -math.Log1p(-xi)
			}
			sumlog1p += tmp
		}
		yeoJohnsonTransform(xTrans, x, lmbda)
		loglike := -float64(nSamples)/2*math.Log(NumpyLike{}.Var(mat.NewDense(nSamples, 1, xTrans)).At(0, 0)) +
			(lmbda-1)*sumlog1p
		return -loglike
	}
	maxFev := func(n int) bool { return n >= 500 }
	brent := optimize.NewBrentMinimizer(negLogLikelihood, 1.48e-8, math.MaxInt32, maxFev)

	brent.Brack = []float64{-2, 2}

	res, fx, iter, funcalls := brent.Optimize()
	_, _, _ = fx, iter, funcalls
	return res
}

// yeoJohnsonTransform returns in out transformed input x following Yeo-Johnson transform with parameter lambda
func yeoJohnsonTransform(out, x []float64, lmbda float64) {
	spacing := math.Nextafter(1, 1) - 1

	//# when x >= 0
	if math.Abs(lmbda) < spacing {
		for pos := range x {
			if x[pos] >= 0 {
				out[pos] = math.Log1p(x[pos])
			}
		}
	} else { // # lmbda != 0
		for pos := range x {
			if x[pos] >= 0 {
				out[pos] = (math.Pow(x[pos]+1, lmbda) - 1) / lmbda
			}
		}
	}
	//# when x < 0
	if math.Abs(lmbda-2) > spacing {
		for pos := range x {
			if x[pos] < 0 {

				out[pos] = -(math.Pow(-x[pos]+1, 2-lmbda) - 1) / (2 - lmbda)
			}
		}
	} else { // # lmbda == 2
		for pos := range x {
			if x[pos] < 0 {
				out[pos] = -math.Log1p(-x[pos])
			}
		}
	}
}

// yeoJohnsonInverseTransform returns inverse-transformed input x following Yeo-Johnson inverse transform with parameter lambda
func yeoJohnsonInverseTransform(xinv, x []float64, lmbda float64) {
	spacing := math.Nextafter(1, 1) - 1
	abs := math.Abs
	exp := math.Exp
	power := math.Pow
	// # when x >= 0
	if abs(lmbda) < spacing {
		for pos := range x {
			if x[pos] >= 0 {
				xinv[pos] = exp(x[pos]) - 1
			}
		}

	} else { // # lmbda != 0
		for pos := range x {
			if x[pos] >= 0 {
				xinv[pos] = power(x[pos]*lmbda+1, 1/lmbda) - 1
			}
		}

	}
	// # when x < 0
	if abs(lmbda-2) > spacing {
		for pos := range x {
			if x[pos] < 0 {
				xinv[pos] = 1 - power(-(2-lmbda)*x[pos]+1, 1/(2-lmbda))
			}
		}

	} else { // # lmbda == 2
		for pos := range x {
			if x[pos] < 0 {
				xinv[pos] = 1 - exp(-x[pos])
			}
		}
	}
}

func boxCoxOptimize(x []float64) float64 {
	xTrans := make([]float64, len(x))
	negLogLikelihood := func(lmbda float64) float64 { // Return the negative log likelihood of the observed data x as a function of lambda
		nSamples := len(x)
		var sumlog float64
		for _, xi := range x {
			sumlog += math.Log(xi)
		}
		boxCoxTransform(xTrans, x, lmbda)
		loglike := -float64(nSamples)/2*math.Log(NumpyLike{}.Var(mat.NewDense(nSamples, 1, xTrans)).At(0, 0)) +
			(lmbda-1)*sumlog
		return -loglike
	}
	maxFev := func(n int) bool { return n >= 500 }
	brent := optimize.NewBrentMinimizer(negLogLikelihood, 1.48e-8, math.MaxInt32, maxFev)

	brent.Brack = []float64{-2, 2}

	res, fx, iter, funcalls := brent.Optimize()
	_, _, _ = fx, iter, funcalls
	return res
}
func boxCoxTransform(out, x []float64, lmbda float64) {
	var f = func(x float64) float64 { return (math.Pow(x, lmbda) - 1) / lmbda }
	if lmbda == 0 {
		f = math.Log
	}
	for pos, xi := range x {
		if xi > 0 || (xi >= 0 && lmbda > 0) {
			out[pos] = f(xi)
		} else {
			out[pos] = math.NaN()
		}
	}
}
func boxCoxInverseTransform(out, x []float64, lmbda float64) {
	var f = func(x float64) float64 { return math.Pow(x*lmbda+1, 1/lmbda) }
	if lmbda == 0 {
		f = math.Exp
	}
	for pos, xi := range x {
		if xi > 0 || (xi >= 0 && lmbda > 0) {
			out[pos] = f(xi)
		} else {
			out[pos] = math.NaN()
		}
	}
}
