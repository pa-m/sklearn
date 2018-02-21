package preprocessing

import (
	"fmt"
	"github.com/gonum/floats"
	"math"
)

type float = float64

// MinMaxScaler rescale data between FeatureRange
type MinMaxScaler struct {
	FeatureRange                            []float
	Scale, Min, DataMin, DataMax, DataRange []float
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
func (scaler *MinMaxScaler) Fit(X [][]float, y []float) *MinMaxScaler {
	return scaler.Reset().PartialFit(X, y)
}

// PartialFit updates Scale and Min with partial data
func (scaler *MinMaxScaler) PartialFit(X [][]float, y []float) *MinMaxScaler {
	if len(X) == 0 {
		return scaler
	}
	nFeatures := len(X[0])
	if scaler.NSamplesSeen == 0 {
		scaler.DataRange = make([]float, nFeatures, nFeatures)
		scaler.DataMin = make([]float, nFeatures, nFeatures)
		scaler.DataMax = make([]float, nFeatures, nFeatures)
		scaler.Min = make([]float, nFeatures, nFeatures)
		scaler.Scale = make([]float, nFeatures, nFeatures)

		copy(scaler.DataMin, X[0])
		copy(scaler.DataMax, X[0])
	}
	for _, Xi := range X {
		for j, Xij := range Xi {
			scaler.DataMin[j] = math.Min(scaler.DataMin[j], Xij)
			scaler.DataMax[j] = math.Max(scaler.DataMax[j], Xij)
		}
		scaler.NSamplesSeen++
	}
	// dataRange = dataMax - dataMin
	floats.SubTo(scaler.DataRange, scaler.DataMax, scaler.DataMin)
	// handle HandleZerosInScale
	//scaler.scale_ = ((featureRange[1] - featureRange[0]) /HandleZerosInScale(dataRange))
	for j, drj := range scaler.DataRange {
		if drj == 0. {
			drj = 1.
		}
		scaler.Scale[j] = (scaler.FeatureRange[1] - scaler.FeatureRange[0]) / drj
	}
	//scaler.min_ = featureRange[0] - dataMin * scaler.scale_
	copy(scaler.Min, scaler.DataMin)
	floats.Mul(scaler.Min, scaler.Scale)
	floats.Scale(-1., scaler.Min)
	floats.AddConst(scaler.FeatureRange[0], scaler.Min)
	fmt.Println("ok")
	return scaler
}

// Transform applies scaling to X
func (scaler *MinMaxScaler) Transform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Mul(Xout[i], scaler.Scale)
		floats.Add(Xout[i], scaler.Min)
	}
	return
}

// InverseTransform rescale data into original bounds
func (scaler *MinMaxScaler) InverseTransform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Sub(Xout[i], scaler.Min)
		floats.Div(Xout[i], scaler.Scale)

	}
	return
}

// StandardScaler scales data by removing Mean and dividing by stddev
type StandardScaler struct {
	Scale, Mean, Var []float
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
func (scaler *StandardScaler) Fit(X [][]float, y []float) *StandardScaler {
	return scaler.Reset().PartialFit(X, y)
}

// PartialFit computes Mean snd Std
func (scaler *StandardScaler) PartialFit(X [][]float, y []float) *StandardScaler {
	if len(X) == 0 {
		return scaler
	}
	nFeatures := len(X[0])
	if scaler.NSamplesSeen == 0 {
		scaler.Var = make([]float, nFeatures, nFeatures)
		scaler.Mean = make([]float, nFeatures, nFeatures)
		scaler.Scale = make([]float, nFeatures, nFeatures)

	}
	scaler.Mean, scaler.Var, scaler.NSamplesSeen = IncrementalMeanAndVar(X, scaler.Mean, scaler.Var, scaler.NSamplesSeen)
	for j, vj := range scaler.Var {
		if vj == 0. {
			scaler.Scale[j] = 1.
		} else {
			scaler.Scale[j] = math.Sqrt(vj)
		}

	}
	return scaler
}

// Transform scales data
func (scaler *StandardScaler) Transform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Sub(Xout[i], scaler.Mean)
		floats.Div(Xout[i], scaler.Scale)
	}
	return
}

// InverseTransform unscales data
func (scaler *StandardScaler) InverseTransform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Mul(Xout[i], scaler.Scale)
		floats.Add(Xout[i], scaler.Mean)
	}
	return
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
func IncrementalMeanAndVar(X [][]float, lastMean, lastVariance []float,
	lastSampleCount int) (updatedMean, updatedVariance []float, updatedSampleCount int) {
	// # old = stats until now
	// # new = the current increment
	// # updated = the aggregated stats
	//lastSum := lastMean * lastSampleCount
	lastSum := make([]float, len(X[0]), len(X[0]))
	copy(lastSum, lastMean)
	floats.Scale(float(lastSampleCount), lastSum)
	// new sum
	newSum := make([]float, len(X[0]), len(X[0]))
	for _, Xi := range X {
		floats.Add(newSum, Xi)
	}

	newSampleCount := len(X)
	updatedSampleCount = lastSampleCount + newSampleCount

	//updatedMean = (lastSum + newSum) / updatedSampleCount
	updatedMean = make([]float, len(X[0]), len(X[0]))
	copy(updatedMean, lastSum)
	floats.Add(updatedMean, newSum)
	floats.Scale(1./float(updatedSampleCount), updatedMean)

	//newUnnormalizedVariance = X.var(axis=0) * newSampleCount
	newUnnormalizedVariance := make([]float, len(X[0]), len(X[0]))
	var updatedUnnormalizedVariance []float
	newMean := make([]float, len(X[0]), len(X[0]))
	copy(newMean, newSum)
	floats.Scale(1./float(newSampleCount), newMean)
	tmp := make([]float, len(X[0]), len(X[0]))
	for _, Xi := range X {
		floats.SubTo(tmp, Xi, newMean)
		floats.Mul(tmp, tmp)
		floats.Add(newUnnormalizedVariance, tmp)
	}

	if lastSampleCount == 0 { //# Avoid division by 0
		updatedUnnormalizedVariance = newUnnormalizedVariance
	} else {
		lastOverNewCount := float(lastSampleCount) / float(newSampleCount)
		//lastUnnormalizedVariance := lastVariance * lastSampleCount
		lastUnnormalizedVariance := make([]float, len(X[0]), len(X[0]))
		copy(lastUnnormalizedVariance, lastVariance)
		floats.Scale(float(lastSampleCount), lastUnnormalizedVariance)

		// updatedUnnormalizedVariance = (
		//     lastUnnormalizedVariance +
		//     newUnnormalizedVariance +
		//     lastOverNewCount / updatedSampleCount *
		//     (lastSum / lastOverNewCount - newSum) ** 2)
		updatedUnnormalizedVariance = make([]float, len(X[0]), len(X[0]))
		tmp := make([]float, len(X[0]), len(X[0]))
		copy(tmp, lastSum)
		floats.Scale(1./lastOverNewCount, tmp)
		floats.Add(tmp, newSum)
		floats.Mul(tmp, tmp)
		floats.Scale(lastOverNewCount/float(updatedSampleCount), tmp)

		copy(updatedUnnormalizedVariance, lastUnnormalizedVariance)
		floats.Add(updatedUnnormalizedVariance, newUnnormalizedVariance)
		floats.Add(updatedUnnormalizedVariance, tmp)
	}
	//updatedVariance = updatedUnnormalizedVariance / updatedSampleCount
	updatedVariance = make([]float, len(X[0]), len(X[0]))
	copy(updatedVariance, updatedUnnormalizedVariance)
	floats.Scale(1./float(updatedSampleCount), updatedVariance)

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
func (scaler *PolynomialFeatures) Fit(X [][]float, y []float) *PolynomialFeatures {
	nFeatures := len(X[0])
	scaler.Powers = make([][]int, 0)
	ppowers := make([]int, nFeatures, nFeatures)
	addpowers(&scaler.Powers, 0, scaler.Degree, nFeatures, ppowers, scaler.InteractionOnly, scaler.IncludeBias)
	return scaler
}

// Transform returns data with polynomial features added
func (scaler *PolynomialFeatures) Transform(X [][]float) [][]float {
	Xout := make([][]float, len(X), len(X))
	for isample, Xi := range X {
		Xout[isample] = make([]float, len(scaler.Powers), len(scaler.Powers))
		for ioutput, p := range scaler.Powers {
			v := 1.
			for j, pj := range p {
				v *= math.Pow(Xi[j], float(pj))
			}
			Xout[isample][ioutput] = v
		}
	}
	return Xout
}
