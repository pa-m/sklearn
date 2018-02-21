package preprocessing

import (
	"fmt"
	"github.com/gonum/floats"
	"math"
)

type float = float64

type MinMaxScaler struct {
	FeatureRange                            []float
	Scale, Min, DataMin, DataMax, DataRange []float
	NSamplesSeen                            int
}

func NewMinMaxScaler(feature_range []float) *MinMaxScaler {
	return &MinMaxScaler{FeatureRange: feature_range}
}

func (self *MinMaxScaler) Reset() *MinMaxScaler {
	self.NSamplesSeen = 0
	return self
}

func (self *MinMaxScaler) Fit(X [][]float, y []float) *MinMaxScaler {
	return self.Reset().PartialFit(X, y)
}

func (self *MinMaxScaler) PartialFit(X [][]float, y []float) *MinMaxScaler {
	if len(X) == 0 {
		return self
	}
	n_features := len(X[0])
	if self.NSamplesSeen == 0 {
		self.DataRange = make([]float, n_features, n_features)
		self.DataMin = make([]float, n_features, n_features)
		self.DataMax = make([]float, n_features, n_features)
		self.Min = make([]float, n_features, n_features)
		self.Scale = make([]float, n_features, n_features)

		copy(self.DataMin, X[0])
		copy(self.DataMax, X[0])
	}
	for _, Xi := range X {
		for j, Xij := range Xi {
			self.DataMin[j] = math.Min(self.DataMin[j], Xij)
			self.DataMax[j] = math.Max(self.DataMax[j], Xij)
		}
		self.NSamplesSeen++
	}
	// data_range = data_max - data_min
	floats.SubTo(self.DataRange, self.DataMax, self.DataMin)
	// handle _handle_zeros_in_scale
	//self.scale_ = ((feature_range[1] - feature_range[0]) /_handle_zeros_in_scale(data_range))
	for j, drj := range self.DataRange {
		if drj == 0. {
			drj = 1.
		}
		self.Scale[j] = (self.FeatureRange[1] - self.FeatureRange[0]) / drj
	}
	//self.min_ = feature_range[0] - data_min * self.scale_
	copy(self.Min, self.DataMin)
	floats.Mul(self.Min, self.Scale)
	floats.Scale(-1., self.Min)
	floats.AddConst(self.FeatureRange[0], self.Min)
	fmt.Println("ok")
	return self
}

func (self *MinMaxScaler) Transform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Mul(Xout[i], self.Scale)
		floats.Add(Xout[i], self.Min)
	}
	return
}

func (self *MinMaxScaler) InverseTransform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Sub(Xout[i], self.Min)
		floats.Div(Xout[i], self.Scale)

	}
	return
}

type StandardScaler struct {
	Scale, Mean, Var []float
	NSamplesSeen     int
}

func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

func (self *StandardScaler) Reset() *StandardScaler {
	self.NSamplesSeen = 0
	return self
}

func (self *StandardScaler) Fit(X [][]float, y []float) *StandardScaler {
	return self.Reset().PartialFit(X, y)
}

func (self *StandardScaler) PartialFit(X [][]float, y []float) *StandardScaler {
	if len(X) == 0 {
		return self
	}
	n_features := len(X[0])
	if self.NSamplesSeen == 0 {
		self.Var = make([]float, n_features, n_features)
		self.Mean = make([]float, n_features, n_features)
		self.Scale = make([]float, n_features, n_features)

	}
	self.Mean, self.Var, self.NSamplesSeen = _incremental_mean_and_var(X, self.Mean, self.Var, self.NSamplesSeen)
	for j, vj := range self.Var {
		if vj == 0. {
			self.Scale[j] = 1.
		} else {
			self.Scale[j] = math.Sqrt(vj)
		}

	}
	return self
}

func (self *StandardScaler) Transform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Sub(Xout[i], self.Mean)
		floats.Div(Xout[i], self.Scale)
	}
	return
}

func (self *StandardScaler) InverseTransform(X [][]float) (Xout [][]float) {
	Xout = make([][]float, len(X), len(X))
	for i, Xi := range X {
		Xout[i] = make([]float, len(Xi), len(Xi))
		copy(Xout[i], Xi)
		floats.Mul(Xout[i], self.Scale)
		floats.Add(Xout[i], self.Mean)
	}
	return
}

// """Calculate mean update and a Youngs and Cramer variance update.
// last_mean and last_variance are statistics computed at the last step by the
// function. Both must be initialized to 0.0. In case no scaling is required
// last_variance can be None. The mean is always required and returned because
// necessary for the calculation of the variance. last_n_samples_seen is the
// number of samples encountered until now.
// From the paper "Algorithms for computing the sample variance: analysis and
// recommendations", by Chan, Golub, and LeVeque.
// Parameters
// ----------
// X : array-like, shape (n_samples, n_features)
//     Data to use for variance update
// last_mean : array-like, shape: (n_features,)
// last_variance : array-like, shape: (n_features,)
// last_sample_count : int
// Returns
// -------
// updated_mean : array, shape (n_features,)
// updated_variance : array, shape (n_features,)
//     If None, only mean is computed
// updated_sample_count : int
// References
// ----------
// T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
//     variance: recommendations, The American Statistician, Vol. 37, No. 3,
//     pp. 242-247
// Also, see the sparse implementation of this in
// `utils.sparsefuncs.incr_mean_variance_axis` and
// `utils.sparsefuncs_fast.incr_mean_variance_axis0`
// """
func _incremental_mean_and_var(X [][]float, last_mean, last_variance []float,
	last_sample_count int) (updated_mean, updated_variance []float, updated_sample_count int) {
	// # old = stats until now
	// # new = the current increment
	// # updated = the aggregated stats
	//last_sum := last_mean * last_sample_count
	last_sum := make([]float, len(X[0]), len(X[0]))
	copy(last_sum, last_mean)
	floats.Scale(float(last_sample_count), last_sum)
	// new sum
	new_sum := make([]float, len(X[0]), len(X[0]))
	for _, Xi := range X {
		floats.Add(new_sum, Xi)
	}

	new_sample_count := len(X)
	updated_sample_count = last_sample_count + new_sample_count

	//updated_mean = (last_sum + new_sum) / updated_sample_count
	updated_mean = make([]float, len(X[0]), len(X[0]))
	copy(updated_mean, last_sum)
	floats.Add(updated_mean, new_sum)
	floats.Scale(1./float(updated_sample_count), updated_mean)

	//new_unnormalized_variance = X.var(axis=0) * new_sample_count
	new_unnormalized_variance := make([]float, len(X[0]), len(X[0]))
	var updated_unnormalized_variance []float
	new_mean := make([]float, len(X[0]), len(X[0]))
	copy(new_mean, new_sum)
	floats.Scale(1./float(new_sample_count), new_mean)
	tmp := make([]float, len(X[0]), len(X[0]))
	for _, Xi := range X {
		floats.SubTo(tmp, Xi, new_mean)
		floats.Mul(tmp, tmp)
		floats.Add(new_unnormalized_variance, tmp)
	}

	if last_sample_count == 0 { //# Avoid division by 0
		updated_unnormalized_variance = new_unnormalized_variance
	} else {
		last_over_new_count := float(last_sample_count) / float(new_sample_count)
		//last_unnormalized_variance := last_variance * last_sample_count
		last_unnormalized_variance := make([]float, len(X[0]), len(X[0]))
		copy(last_unnormalized_variance, last_variance)
		floats.Scale(float(last_sample_count), last_unnormalized_variance)

		// updated_unnormalized_variance = (
		//     last_unnormalized_variance +
		//     new_unnormalized_variance +
		//     last_over_new_count / updated_sample_count *
		//     (last_sum / last_over_new_count - new_sum) ** 2)
		updated_unnormalized_variance = make([]float, len(X[0]), len(X[0]))
		tmp := make([]float, len(X[0]), len(X[0]))
		copy(tmp, last_sum)
		floats.Scale(1./last_over_new_count, tmp)
		floats.Add(tmp, new_sum)
		floats.Mul(tmp, tmp)
		floats.Scale(last_over_new_count/float(updated_sample_count), tmp)

		copy(updated_unnormalized_variance, last_unnormalized_variance)
		floats.Add(updated_unnormalized_variance, new_unnormalized_variance)
		floats.Add(updated_unnormalized_variance, tmp)
	}
	//updated_variance = updated_unnormalized_variance / updated_sample_count
	updated_variance = make([]float, len(X[0]), len(X[0]))
	copy(updated_variance, updated_unnormalized_variance)
	floats.Scale(1./float(updated_sample_count), updated_variance)

	return updated_mean, updated_variance, updated_sample_count
}

type PolynomialFeatures struct {
	Degree                       int
	InteractionOnly, IncludeBias bool
	Powers                       [][]int
}

func NewPolynomialFeatures() *PolynomialFeatures {
	return &PolynomialFeatures{Degree: 2, InteractionOnly: false, IncludeBias: true}

}

func addpowers(Powers *[][]int, j, Degree, n_features int, ppowers []int, InteractionOnly, IncludeBias bool) {
	ppsum := 0
	for jj := 0; jj < j; jj++ {
		ppsum += ppowers[jj]
	}

	for d := 0; d <= Degree-ppsum; d++ {
		ppowers[j] = d
		if j < n_features-1 {

			addpowers(Powers, j+1, Degree, n_features, ppowers, InteractionOnly, IncludeBias)
		} else {
			if !IncludeBias && ppsum+d == 0 {
				continue
			}
			if InteractionOnly {
				nnotz := 0
				for j1 := 0; j1 < n_features; j1++ {
					if ppowers[j1] != 0 {
						nnotz++
					}
				}
				if nnotz > 1 {
					continue
				}
			}
			//fmt.Printf("append %v\n", ppowers)
			ppower := make([]int, n_features, n_features)
			copy(ppower, ppowers)

			*Powers = append(
				*Powers, ppower)
		}
	}
}

func (self *PolynomialFeatures) Fit(X [][]float, y []float) *PolynomialFeatures {
	n_features := len(X[0])
	//powers_[i, j] is the exponent of the jth input in the ith output.
	self.Powers = make([][]int, 0)
	ppowers := make([]int, n_features, n_features)
	addpowers(&self.Powers, 0, self.Degree, n_features, ppowers, self.InteractionOnly, self.IncludeBias)
	return self
}

func (self *PolynomialFeatures) Transform(X [][]float) [][]float {
	Xout := make([][]float, len(X), len(X))
	for isample, Xi := range X {
		Xout[isample] = make([]float, len(self.Powers), len(self.Powers))
		for ioutput, p := range self.Powers {
			v := 1.
			for j, pj := range p {
				v *= math.Pow(Xi[j], float(pj))
			}
			Xout[isample][ioutput] = v
		}
	}
	return Xout
}
