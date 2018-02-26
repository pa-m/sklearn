package base

import (
	"math"
)

// GD contains data for Gradient Descent regressor
type GD struct {
	RegressorMixin1
	Epochs                                             int
	LearningRate, Decay, Tol, Momentum, Alpha, L1Ratio float
	Coefs                                              []float
}

// NewGD create a GD with reasonable defaults
func NewGD() *GD {
	self := &GD{Epochs: 3000, LearningRate: 1e-3, Decay: .95, Tol: 1e-3, Momentum: .5}
	self.Predicter = self
	return self
}

// Fit learns GD Coefs
// adapted from gdSolver from  https://github.com/ohheydom/linearregression/blob/master/linear_regression.go
// Gradient Descent algorithm.
func (gd *GD) Fit(x [][]float64, y []float64) *GD {
	n, nFeatures := len(x), len(x[0])
	gamma := gd.LearningRate / float(n)
	w := make([]float64, nFeatures+1)
	dw := make([]float64, nFeatures+1)
	gd.Coefs = w
	errors := make([]float64, n)
	for i := 0; i < gd.Epochs; i++ {
		Shuffle(x, y)
		predY := gd.Predict(x)
		errorSum2 := 0.0
		// compute errorSum for updating w[0] and errorSum2 for Tol check
		for k := 0; k < n; k++ {
			errors[k] = y[k] - predY[k]
			errorSum2 += math.Pow(errors[k], 2.)
		}
		//fmt.Printf("GD %v %v\n", w, errorSum)
		// update w[1]..w[n]
		for k := 0; k < n; k++ {
			dw[0] = gd.Momentum*dw[0] + gamma*errors[k]
			w[0] += dw[0]
			for l := 1; l < len(w); l++ {
				// MomentumPart:=gd.Momentum*dw[l]
				// ErrorPart:=gamma*errors[k]*x[k][l-1]
				// RegulPart := gd.Alpha/float(n)*w[l]
				dw[l] = gd.Momentum*dw[l] + gamma*x[k][l-1]*errors[k] + gd.Alpha/float(n)*w[l]
				w[l] += dw[l]
			}
		}
		// L1 : floats.sum(ewize(w,math.Abs));L2:=sum(ewise(w,func(w float)float{return w*w}));R=gd.L1Ratio*L1+(1-gd.L1Ratio*L2)
		// TODO: use L1Ratio

		//decrease lr/n
		// TODO learning_rate=optimal eta(t)=1/(alpha*(t0+t))
		gamma *= gd.Decay
		//fmt.Printf("Epoch %d gamma:%g\n", i, gamma)
		//check Tol
		if math.Sqrt(errorSum2)/float(n) < gd.Tol {
			break
		}
	}
	gd.Coefs = w
	return gd
}

// Predict uses the GD Coefs to calculate each sample's label.
func (gd *GD) Predict(x [][]float64) []float64 {
	w := gd.Coefs
	n, nFeatures := len(x), len(x[0])
	predY := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 1; j <= nFeatures; j++ {
			predY[i] += x[i][j-1] * w[j]
		}
		predY[i] += w[0]
	}
	return predY
}
