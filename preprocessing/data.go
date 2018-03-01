package preprocessing

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type float = float64

// Transformer is an interface for various preprocessors
type Transformer interface {
	Fit(X, Y *mat.Dense) Transformer
	Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense)
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

// InverseTransform rescale data into original bounds
func (scaler *MinMaxScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, nFeatures := X.Dims()
	Xout = mat.NewDense(nSamples, nFeatures, nil)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return (x - scaler.Min.At(0, j)) / scaler.Scale.At(0, j)
	}, X)
	return Xout, Y
}

// StandardScaler scales data by removing Mean and dividing by stddev
type StandardScaler struct {
	Scale, Mean, Var *mat.Dense
	NSamplesSeen     int
}

// NewStandardScaler creates a *StandardScaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
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
	Xout = mat.DenseCopyOf(X)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return (x - scaler.Mean.At(0, j)) / scaler.Scale.At(0, j)
	}, X)
	return Xout, Y
}

// InverseTransform unscales data
func (scaler *StandardScaler) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout = mat.DenseCopyOf(X)
	Xout.Apply(func(i int, j int, x float64) float64 {
		return scaler.Mean.At(0, j) + x*scaler.Scale.At(0, j)
	}, X)
	return Xout, Y
}

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
func NewPolynomialFeatures() *PolynomialFeatures {
	return &PolynomialFeatures{Degree: 2, InteractionOnly: false, IncludeBias: true}

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

// InverseTransform inverse tranformation for PolynomialFeatures. not implemented.
func (scaler *PolynomialFeatures) InverseTransform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	//TODO
	type jt struct{ jorig, jpoly int }
	var jts []jt
	nSamples, _ := X.Dims()
	for ioutput, p := range scaler.Powers {
		var jWith1 = -1
		var sumpj = 0
		for j, pj := range p {
			sumpj = +pj
			if pj == 1 {
				jWith1 = j
			}
		}
		if sumpj == 1 && jWith1 > 0 {
			jts = append(jts, jt{jWith1, ioutput})
		}
	}
	Xout = mat.NewDense(nSamples, len(jts), nil)
	for _, pair := range jts {
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

	nSamples, nFeatures := X.Dims()
	XOffset = mat.NewDense(1, nFeatures, nil)
	if FitIntercept {
		DenseMean(XOffset, X)
	}
	XScale = mat.NewDense(1, nFeatures, nil)
	XScale.Apply(func(i int, j int, XOffset float64) float64 {
		v := 0.
		if Normalize {
			for i := 0; i < nSamples; i++ {
				v += math.Pow(X.At(i, j)-XOffset, 2)
			}
			v = math.Sqrt(v / float64(nSamples))
		}
		if v == 0. {
			v = 1.
		}
		return v
	}, XOffset)
	X.Apply(func(i int, j int, v float64) float64 {
		return (v - XOffset.At(0, j)) / XScale.At(0, j)
	}, X)
	return XOffset, XScale
}
