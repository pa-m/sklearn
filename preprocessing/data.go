package preprocessing

import (
	"math"
	"math/rand"
	"sort"

	"github.com/pa-m/sklearn/base"

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

// Reset resets scaler to its initial state
func (scaler *MinMaxScaler) Reset() *MinMaxScaler {
	scaler.NSamplesSeen = 0
	return scaler
}

// Fit computes Sale and Min
func (scaler *MinMaxScaler) Fit(X, Y *mat.Dense) Transformer {
	scaler.Reset()
	return scaler.PartialFit(X, Y)
}

// PartialFit updates Scale and Min with partial data
func (scaler *MinMaxScaler) PartialFit(X, Y *mat.Dense) Transformer {
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

		scaler.DataMin.Clone(X.RowView(0).T())
		scaler.DataMax.Clone(X.RowView(0).T())
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
func (scaler *MinMaxScaler) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return scaler.Min.At(0, j) + x*scaler.Scale.At(0, j)
	}, X)
	return Xout, Y
}

// FitTransform for MinMaxScaler
func (scaler *MinMaxScaler) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return scaler.Fit(X, Y).Transform(X, Y)
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

// Reset ...
func (scaler *StandardScaler) Reset() *StandardScaler {
	scaler.NSamplesSeen = 0
	return scaler
}

// Fit computes Mean snd Std
func (scaler *StandardScaler) Fit(X, Y *mat.Dense) Transformer {
	scaler.Reset()
	return scaler.PartialFit(X, Y)
}

// PartialFit computes Mean snd Std
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
func (scaler *StandardScaler) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
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

			Xoutmat.Data[jXout+i] = (Xmat.Data[jX+i] - mean) / scale
		}
	}
	return Xout, Y
}

// FitTransform for StandardScaler
func (scaler *StandardScaler) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return scaler.Fit(X, Y).Transform(X, Y)
}

// InverseTransform unscales data
func (scaler *StandardScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
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
		tmp := make([]float64, m, m)
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
		tmp := make([]float64, m, m)
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
		tmp := make([]float64, m, m)
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
func (scaler *RobustScaler) Fit(X, Y *mat.Dense) Transformer {
	scaler.Reset()
	return scaler.PartialFit(X, Y)
}

// PartialFit computes Median and Quantiles
func (scaler *RobustScaler) PartialFit(X, Y *mat.Dense) Transformer {
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
func (scaler *RobustScaler) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
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
	return Xout, Y
}

// FitTransform for RobustScaler
func (scaler *RobustScaler) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return scaler.Fit(X, Y).Transform(X, Y)
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
		updatedUnnormalizedVariance.Clone(newUnnormalizedVariance)
	} else {
		lastOverNewCount := float(lastSampleCount) / float(newSampleCount)
		//lastUnnormalizedVariance := lastVariance * lastSampleCount
		lastUnnormalizedVariance := mat.NewDense(1, nFeatures, nil)
		lastUnnormalizedVariance.Clone(lastVariance)
		lastUnnormalizedVariance.Scale(float(lastSampleCount), lastUnnormalizedVariance)

		// updatedUnnormalizedVariance = (
		//     lastUnnormalizedVariance +
		//     newUnnormalizedVariance +
		//     lastOverNewCount / updatedSampleCount *
		//     (lastSum / lastOverNewCount - newSum) ** 2)
		tmp.Clone(lastSum)
		tmp.Scale(1./lastOverNewCount, tmp)
		tmp.Add(tmp, newSum)
		tmp.Mul(tmp, tmp)
		tmp.Scale(lastOverNewCount/float(updatedSampleCount), tmp)

		updatedUnnormalizedVariance.Clone(lastUnnormalizedVariance)
		updatedUnnormalizedVariance.Add(updatedUnnormalizedVariance, newUnnormalizedVariance)
		updatedUnnormalizedVariance.Add(updatedUnnormalizedVariance, tmp)
	}
	//updatedVariance = updatedUnnormalizedVariance / updatedSampleCount
	updatedVariance = mat.NewDense(1, nFeatures, nil)
	updatedVariance.Clone(updatedUnnormalizedVariance)
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

func addpowers(Powers *[][]int, j, Degree, nFeatures int, ppowers []int, InteractionOnly, IncludeBias bool) {
	ppsum := 0
	for jj := 0; jj < j; jj++ {
		ppsum += ppowers[jj]
	}

	for d := 0; d <= Degree-ppsum; d++ {
		ppowers[j] = d
		if j < nFeatures-1 {

			addpowers(Powers, j+1, Degree, nFeatures, ppowers, InteractionOnly, IncludeBias)
		} else {
			if !IncludeBias && ppsum+d == 0 {
				continue
			}
			if InteractionOnly {
				nnotz := 0
				for j1 := 0; j1 < nFeatures; j1++ {
					if ppowers[j1] != 0 {
						nnotz++
					}
				}
				if nnotz > 1 {
					continue
				}
			}
			//fmt.Printf("append %v\n", ppowers)
			ppower := make([]int, nFeatures, nFeatures)
			copy(ppower, ppowers)

			*Powers = append(
				*Powers, ppower)
		}
	}
}

// Fit precompute Powers
// Powers[i, j] is the exponent of the jth input in the ith output.
func (scaler *PolynomialFeatures) Fit(X, Y *mat.Dense) Transformer {
	_, nFeatures := X.Dims()
	scaler.Powers = make([][]int, 0)
	ppowers := make([]int, nFeatures, nFeatures)
	addpowers(&scaler.Powers, 0, scaler.Degree, nFeatures, ppowers, scaler.InteractionOnly, scaler.IncludeBias)
	return scaler
}

// Transform returns data with polynomial features added
func (scaler *PolynomialFeatures) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	Xout = mat.NewDense(nSamples, len(scaler.Powers), nil)
	for i := 0; i < nSamples; i++ {
		for ioutput, p := range scaler.Powers {
			v := 1.
			for j, pj := range p {
				v *= math.Pow(X.At(i, j), float(pj))
			}
			Xout.Set(i, ioutput, v)
		}
	}
	return Xout, Y
}

// FitTransform for PolynomialFeatures
func (scaler *PolynomialFeatures) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return scaler.Fit(X, Y).Transform(X, Y)
}

// InverseTransform inverse tranformation for PolynomialFeatures.
func (scaler *PolynomialFeatures) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
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
	for ioutput, p := range scaler.Powers {
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

// DenseNormalize normalize matrix rows by removing mean and dividing with standard deviation
func DenseNormalize(X *mat.Dense, FitIntercept, Normalize bool) (XOffset, XScale *mat.Dense) {
	scaler := NewStandardScaler()
	scaler.WithMean = FitIntercept
	scaler.WithStd = Normalize
	Xout, _ := scaler.FitTransform(X, nil)
	X.Copy(Xout)

	fill := func(M *mat.Dense, v float64) {
		Mmat := M.RawMatrix()
		for jM := 0; jM < Mmat.Rows*Mmat.Stride; jM = jM + Mmat.Stride {
			for i := range Mmat.Data[jM : jM+Mmat.Cols] {
				Mmat.Data[jM+i] = v
			}
		}
	}
	if !FitIntercept {
		fill(scaler.Mean, 0.)
	}
	if !Normalize {
		fill(scaler.Scale, 1.)
	}
	return scaler.Mean, scaler.Scale
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
	X.Clone(X1)
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

// Fit ...
func (m *OneHotEncoder) Fit(X, Y *mat.Dense) Transformer {
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
func (m *OneHotEncoder) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, nfeatures := X.Dims()
	Yout = Y
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

// FitTransform for OneHotEncoder
func (m *OneHotEncoder) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return m.Fit(X, Y).Transform(X, Y)
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
type Shuffler struct{ Perm []int }

// NewShuffler returns a *Shuffler
func NewShuffler() *Shuffler { return &Shuffler{} }

// Fit for Shuffler
func (m *Shuffler) Fit(X, Y *mat.Dense) Transformer {
	m.Perm = rand.Perm(X.RawMatrix().Rows)
	return m
}

// Transform for Shuffler
func (m *Shuffler) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	xmat, ymat := X.RawMatrix(), Y.RawMatrix()
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
	return X, Y
}

// FitTransform for Shuffler
func (m *Shuffler) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	return m.Fit(X, Y).Transform(X, Y)
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

// Fit for binarizer does nothing
func (m *Binarizer) Fit(X, Y *mat.Dense) Transformer {
	return m
}

// Transform for Binarizer
func (m *Binarizer) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	rmx := X.RawMatrix()
	Xout = mat.NewDense(rmx.Rows, rmx.Cols, nil)
	rmXout := Xout.RawMatrix()
	Yout = Y
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

// FitTransform for Binarizer
func (m *Binarizer) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}

// MaxAbsScaler ...
type MaxAbsScaler struct {
	Scale, MaxAbs []float64
	NSamplesSeen  int
}

// NewMaxAbsScaler ...
func NewMaxAbsScaler() *MaxAbsScaler { return &MaxAbsScaler{} }

// Fit for MaxAbsScaler ...
func (m *MaxAbsScaler) Fit(X, Y *mat.Dense) base.Transformer {
	Xmat := X.RawMatrix()
	m.MaxAbs = make([]float64, Xmat.Cols, Xmat.Cols)
	m.Scale = make([]float64, Xmat.Cols, Xmat.Cols)
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
func (m *MaxAbsScaler) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xmat := X.RawMatrix()
	Xout = mat.NewDense(Xmat.Rows, Xmat.Cols, nil)
	Xoutmat := Xout.RawMatrix()
	for jX, jXout := 0, 0; jX < Xmat.Rows*Xmat.Stride; jX, jXout = jX+Xmat.Stride, jXout+Xoutmat.Stride {
		for i, v := range Xmat.Data[jX : jX+Xmat.Cols] {
			Xoutmat.Data[jXout+i] = v / m.Scale[i]
		}
	}
	Yout = Y
	return
}

// FitTransform for MaxAbsScaler ...
func (m *MaxAbsScaler) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
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
type Normalizer struct{ Norm string }

// NewNormalizer ...
func NewNormalizer() *Normalizer { return &Normalizer{} }

// Fit for Normalizer ...
func (m *Normalizer) Fit(X, Y *mat.Dense) base.Transformer { return m }

// Transform for Normalizer ...
func (m *Normalizer) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	Xout = mat.NewDense(NSamples, NFeatures, nil)
	var norm = 2.
	switch m.Norm {
	case "l1":
		norm = 1.
	case "max":
		norm = math.Inf(1)
	}
	for i := 0; i < NSamples; i++ {
		v := X.RowView(i)
		floats.AddScaled(Xout.RawRowView(i), 1./mat.Norm(v, norm), v.(mat.RawVectorer).RawVector().Data)

	}
	Yout = Y
	return
}

// FitTransform for Normalizer ...
func (m *Normalizer) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}

// InverseTransform for Normalizer ...
func (m *Normalizer) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = X, Y
	return
}

// KernelCenterer Center a kernel matrix
type KernelCenterer struct {
	KFitAll  float64
	KFitRows []float64
}

// NewKernelCenterer ...
func NewKernelCenterer() *KernelCenterer { return &KernelCenterer{} }

// Fit for KernelCenterer ...
func (m *KernelCenterer) Fit(X, Y *mat.Dense) base.Transformer {
	r, c := X.Dims()
	m.KFitRows = make([]float64, c, c)

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
func (m *KernelCenterer) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	r, _ := X.Dims()
	KPredCols := make([]float64, r, r)
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
	Yout = Y
	return
}

// FitTransform for KernelCenterer ...
func (m *KernelCenterer) FitTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout, Yout = m.Fit(X, Y).Transform(X, Y)
	return
}
