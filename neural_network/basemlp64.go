package neuralnetwork

import (
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	"github.com/pa-m/sklearn/base"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/optimize"
)

// BaseMultilayerPerceptron64 closely matches sklearn/neural_network/multilayer_perceptron.py
type BaseMultilayerPerceptron64 struct {
	Activation         string
	Solver             string
	Alpha              float64
	WeightDecay        float64
	BatchSize          int
	LearningRate       string
	LearningRateInit   float64
	PowerT             float64
	MaxIter            int
	LossFuncName       string
	HiddenLayerSizes   []int
	Shuffle            bool
	RandomState        base.RandomState
	Tol                float64
	Verbose            bool
	WarmStart          bool
	Momentum           float64
	NesterovsMomentum  bool
	EarlyStopping      bool
	ValidationFraction float64
	Beta1              float64
	Beta2              float64
	Epsilon            float64
	NIterNoChange      int

	// Outputs
	NLayers       int
	NIter         int
	NOutputs      int
	Intercepts    [][]float64
	Coefs         []blas64General
	OutActivation string
	Loss          float64

	// internal
	t                   int
	LossCurve           []float64
	ValidationScores    []float64
	BestValidationScore float64
	BestLoss            float64
	NoImprovementCount  int
	InterceptsGrads     [][]float64
	CoefsGrads          []blas64General
	optimizer           Optimizer64
	packedParameters    []float64
	packedGrads         []float64
	bestParameters      []float64
}

// Activations64 is a map containing the inplace_activation functions
var Activations64 = map[string]func(z blas64General){
	"identity": func(z blas64General) {},
	"logistic": func(z blas64General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] = 1 / (1 + M64.Exp(-z.Data[zpos+col]))
			}
		}
	},
	"tanh": func(z blas64General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] = M64.Tanh(-z.Data[zpos+col])
			}
		}
	},
	"relu": func(z blas64General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				if z.Data[zpos+col] < 0 {
					z.Data[zpos+col] = 0
				}
			}
		}
	},
	"softmax": func(z blas64General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			sum := float64(0)
			for col := 0; col < z.Cols; col++ {

				z.Data[zpos+col] = M64.Exp(z.Data[zpos+col])
				sum += z.Data[zpos+col]
			}
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] /= sum
			}
		}
	},
}

// Derivatives64 is a map of functions which multiply deltas with derivative of activation function
var Derivatives64 = map[string]func(Z, deltas blas64General){
	"identity": func(Z, deltas blas64General) {
	},
	"logistic": func(Z, deltas blas64General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				z := Z.Data[zpos+col]
				deltas.Data[dpos+col] *= z * (1 - z)
			}
		}
	},
	"tanh": func(Z, deltas blas64General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				z := Z.Data[zpos+col]
				deltas.Data[dpos+col] *= 1 - z*z
			}
		}
	},
	"relu": func(Z, deltas blas64General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				if Z.Data[zpos+col] == 0 {
					deltas.Data[dpos+col] = 0
				}
			}
		}
	},
}

// LossFunctions64 is a map for loss functions
var LossFunctions64 = map[string]func(y, h blas64General) float64{
	"square_loss": func(y, h blas64General) float64 {
		sum := float64(0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				e := h.Data[hpos+col] - y.Data[ypos+col]
				sum += e * e
			}
		}
		return sum / 2 / float64(h.Rows)
	},
	"log_loss": func(y, h blas64General) float64 {
		sum := float64(0)
		hmin, hmax := M64.Nextafter(0, 1), M64.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				if y.Data[ypos+col] != 0 {
					sum += -y.Data[ypos+col] * M64.Log(hval)
				}
			}
		}
		return sum / float64(h.Rows)
	},
	"binary_log_loss": func(y, h blas64General) float64 {
		sum := float64(0)
		hmin, hmax := M64.Nextafter(0, 1), M64.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				sum += -y.Data[ypos+col]*M64.Log(hval) - (1-y.Data[ypos+col])*M64.Log1p(-hval)
			}
		}
		return sum / float64(h.Rows)
	},
}

// Optimizer64 is an interface for stochastic optimizers
type Optimizer64 interface {
	iterationEnds(timeStep float64)
	triggerStopping(msg string, verbose bool) bool
	updateParams(grads []float64)
}

func addIntercepts64(a blas64General, b []float64) {
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			a.Data[apos+c] += b[c]
		}
	}
}
func matDivFloat64(a blas64General, b float64) {
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			a.Data[apos+c] /= b
		}
	}
}
func matRowMean64(a blas64General, b []float64) {
	for c := 0; c < a.Cols; c++ {
		b[c] = 0
	}
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			b[c] += a.Data[apos+c]
		}
	}
	for c := 0; c < a.Cols; c++ {
		b[c] /= float64(a.Rows)
	}
}
func matSub64(a, b, dst blas64General) {
	for arow, apos, bpos := 0, 0, 0; arow < b.Rows; arow, apos, bpos = arow+1, apos+a.Stride, bpos+b.Stride {
		for c := 0; c < a.Cols; c++ {
			dst.Data[apos+c] = a.Data[apos+c] - b.Data[bpos+c]
		}
	}
}

// NewBaseMultilayerPerceptron64 returns a BaseMultilayerPerceptron64 with defaults
func NewBaseMultilayerPerceptron64() *BaseMultilayerPerceptron64 {
	return &BaseMultilayerPerceptron64{

		Activation:       "relu",
		Solver:           "adam",
		Alpha:            0.0001,
		BatchSize:        200,
		LearningRate:     "constant",
		LearningRateInit: 0.001,
		PowerT:           .5,
		MaxIter:          200,
		//LossFuncName       string
		HiddenLayerSizes: []int{100},
		Shuffle:          true,
		//RandomState       base.Source
		Tol:                1e-4,
		Verbose:            false,
		WarmStart:          false,
		Momentum:           .9,
		NesterovsMomentum:  true,
		EarlyStopping:      false,
		ValidationFraction: .1,
		Beta1:              .9,
		Beta2:              .999,
		Epsilon:            1e-8,
		NIterNoChange:      10,
	}
}

// forwardPass Perform a forward pass on the network by computing the values
// of the neurons in the hidden layers and the output layer.
//        activations : []blas64General, length = nLayers - 1
func (mlp *BaseMultilayerPerceptron64) forwardPass(activations []blas64General) {
	hiddenActivation := Activations64[mlp.Activation]
	var i int
	for i = 0; i < mlp.NLayers-1; i++ {
		gemm64(blas.NoTrans, blas.NoTrans, 1, activations[i], mlp.Coefs[i], 0, activations[i+1])
		addIntercepts64(activations[i+1], mlp.Intercepts[i])
		// For the hidden layers
		if (i + 1) != (mlp.NLayers - 1) {
			hiddenActivation(activations[i+1])
		}
	}
	i = mlp.NLayers - 2
	// # For the last layer
	outputActivation := Activations64[mlp.OutActivation]
	outputActivation(activations[i+1])
}

func (mlp *BaseMultilayerPerceptron64) sumCoefSquares() float64 {
	s := float64(0)
	for _, c := range mlp.Coefs {
		for _, co := range c.Data {
			s += co * co
		}
	}
	return s
}

// computeLossGrad Compute the gradient of loss with respect to coefs and intercept for specified layer.
// This function does backpropagation for the specified one layer.
func (mlp *BaseMultilayerPerceptron64) computeLossGrad(layer, NSamples int, activations []blas64General, deltas []blas64General, coefGrads []blas64General, interceptGrads [][]float64) {
	//coefGrads[layer] = safeSparseDot(activations[layer].T, deltas[layer])

	gemm64(blas.Trans, blas.NoTrans, 1, activations[layer], deltas[layer], 0, coefGrads[layer])
	//coefGrads[layer] += (self.alpha * self.coefs_[layer])
	axpy64(len(coefGrads[layer].Data), mlp.Alpha, mlp.Coefs[layer].Data, coefGrads[layer].Data)
	// coefGrads[layer] /= nSamples
	matDivFloat64(coefGrads[layer], float64(NSamples))
	// interceptGrads[layer] = np.mean(deltas[layer], 0)
	matRowMean64(deltas[layer], interceptGrads[layer])
}

// backprop Compute the MLP loss function and its corresponding derivatives with respect to each parameter: weights and bias vectors.
// X : blas64General shape (nSamples, nFeatures)
// Y : blas64General shape (nSamples, nOutputs)
// activations : []blas64General, length=NLayers-1
// deltas : []blas64General, length=NLayers-1
// coefGrads : []blas64General, length=NLayers-1
// interceptGrads : [][]float64, length=NLayers-1

func (mlp *BaseMultilayerPerceptron64) backprop(X, y blas64General, activations, deltas, coefGrads []blas64General, interceptGrads [][]float64) float64 {
	nSamples := X.Rows
	if mlp.WeightDecay > 0 {
		for iw := range mlp.packedParameters {
			mlp.packedParameters[iw] *= (1 - mlp.WeightDecay)
		}
	}
	mlp.forwardPass(activations)

	//# Get loss
	lossFuncName := mlp.LossFuncName
	if strings.EqualFold(lossFuncName, "log_loss") && strings.EqualFold(mlp.OutActivation, "logistic") {
		lossFuncName = "binary_log_loss"
	}
	// y may have less rows than activations il last batch
	loss := LossFunctions64[lossFuncName](y, activations[len(activations)-1])
	// # Add L2 regularization term to loss
	loss += (0.5 * mlp.Alpha) * mlp.sumCoefSquares() / float64(nSamples)

	//# Backward propagate
	last := mlp.NLayers - 2

	// # The calculation of delta[last] here works with following
	// # combinations of output activation and loss function:
	// # sigmoid and binary cross entropy, softmax and categorical cross
	// # entropy, and identity with squared loss
	//deltas[last] = activations[len(activations)-1] - y

	// y may have less rows than activations il last batch
	matSub64(activations[len(activations)-1], y, deltas[last])

	//# Compute gradient for the last layer
	mlp.computeLossGrad(
		last, nSamples, activations, deltas, coefGrads, interceptGrads)

	//# Iterate over the hidden layers
	for i := mlp.NLayers - 2; i >= 1; i-- {
		//deltas[i - 1] = safeSparseDot(deltas[i], self.coefs_[i].T)
		gemm64(blas.NoTrans, blas.Trans, 1, deltas[i], mlp.Coefs[i], 0, deltas[i-1])

		inplaceDerivative := Derivatives64[mlp.Activation]
		inplaceDerivative(activations[i], deltas[i-1])

		mlp.computeLossGrad(
			i-1, nSamples, activations, deltas, coefGrads,
			interceptGrads)

	}
	return loss
}

func (mlp *BaseMultilayerPerceptron64) initialize(y blas64General, layerUnits []int, isClassifier, isMultiClass bool) {
	// # set all attributes, allocate weights etc for first call
	// # Initialize parameters
	mlp.NIter = 0
	mlp.t = 0
	mlp.NOutputs = y.Cols

	//# Compute the number of layers
	mlp.NLayers = len(layerUnits)

	//# Output for regression
	if !isClassifier {
		mlp.OutActivation = "identity"
		mlp.LossFuncName = "square_loss"
		//# Output for multi class
	} else if isMultiClass {
		mlp.OutActivation = "softmax"
		mlp.LossFuncName = "log_loss"
		//# Output for binary class and multi-label
	} else {
		mlp.OutActivation = "logistic"
		mlp.LossFuncName = "binary_log_loss"
	}
	//# Initialize coefficient and intercept layers
	mlp.Coefs = make([]blas64General, mlp.NLayers-1, mlp.NLayers-1)
	mlp.Intercepts = make([][]float64, mlp.NLayers-1, mlp.NLayers-1)
	mlp.CoefsGrads = make([]blas64General, mlp.NLayers-1, mlp.NLayers-1)
	mlp.InterceptsGrads = make([][]float64, mlp.NLayers-1, mlp.NLayers-1)
	off := 0
	for i := 0; i < mlp.NLayers-1; i++ {
		off += (1 + layerUnits[i]) * layerUnits[i+1]
	}
	mem := make([]float64, 2*off, 2*off)
	mlp.packedParameters = mem[0:off]
	mlp.packedGrads = mem[off : 2*off]

	off = 0
	if mlp.RandomState == (base.RandomState)(nil) {
		mlp.RandomState = base.NewLockedSource(uint64(time.Now().UnixNano()))
	}
	type Float64er interface{ Float64() float64 }
	var rndFloat64 func() float64
	if float64er, ok := mlp.RandomState.(Float64er); ok {
		rndFloat64 = float64er.Float64
	} else {
		rndFloat64 = rand.New(mlp.RandomState).Float64
	}
	for i := 0; i < mlp.NLayers-1; i++ {
		prevOff := off
		mlp.Intercepts[i] = mem[off : off+layerUnits[i+1]]
		off += layerUnits[i+1]
		mlp.Coefs[i] = blas64General{Rows: layerUnits[i], Cols: layerUnits[i+1], Stride: layerUnits[i+1], Data: mem[off : off+layerUnits[i]*layerUnits[i+1]]}
		off += layerUnits[i] * layerUnits[i+1]
		// # Use the initialization method recommended by
		// # Glorot et al.
		factor := float64(6.)
		fanIn, fanOut := layerUnits[i], layerUnits[i+1]
		if strings.EqualFold(mlp.Activation, "logistic") {
			factor = 2.
		}
		initBound := M64.Sqrt(factor / float64(fanIn+fanOut))
		for pos := prevOff; pos < off; pos++ {
			mem[pos] = rndFloat64() * initBound
		}
	}
	for i := 0; i < mlp.NLayers-1; i++ {
		mlp.InterceptsGrads[i] = mem[off : off+layerUnits[i+1]]
		off += layerUnits[i+1]
		mlp.CoefsGrads[i] = blas64General{Rows: layerUnits[i], Cols: layerUnits[i+1], Stride: layerUnits[i+1], Data: mem[off : off+layerUnits[i]*layerUnits[i+1]]}
		off += layerUnits[i] * layerUnits[i+1]
	}

	// if self.solver in STOCHASTICSOLVERS:
	//     self.lossCurve_ = []
	//     self.NoImprovementCount = 0
	//     if self.earlyStopping:
	//         self.validationScores_ = []
	//         self.bestValidationScore_ = -np.inf
	//     else:
	//         self.bestLoss_ = np.inf
	mlp.BestLoss = M64.Inf(1)
}

func (mlp *BaseMultilayerPerceptron64) fit(X, y blas64General, incremental bool) {
	// # Validate input parameters.
	mlp.validateHyperparameters()
	for _, s := range mlp.HiddenLayerSizes {
		if s < 0 {
			log.Panicf("hiddenLayerSizes must be > 0, got %v.", mlp.HiddenLayerSizes)
		}
	}
	X, y = mlp.validateInput(X, y, incremental)
	nSamples, nFeatures := X.Rows, X.Cols

	mlp.NOutputs = y.Cols
	layerUnits := append([]int{nFeatures}, mlp.HiddenLayerSizes...)
	layerUnits = append(layerUnits, mlp.NOutputs)

	if mlp.RandomState == nil {
		mlp.RandomState = rand.New(base.NewLockedSource(uint64(time.Now().UnixNano())))
	}
	if !mlp.WarmStart && !incremental {
		//# First time training the model
		var isClassifier, isMulticlass = true, y.Cols > 1
		for _, yval := range y.Data {
			if yval != 0 && yval != 1 {
				isClassifier = false
				break
			}
		}
		mlp.initialize(y, layerUnits, isClassifier, isMulticlass)
	}

	//    # lbfgs does not support mini-batches
	if strings.EqualFold(mlp.Solver, "lbfgs") {
		mlp.BatchSize = nSamples
	} else if mlp.BatchSize <= 0 {
		mlp.BatchSize = nSamples
		if mlp.BatchSize > 200 {
			mlp.BatchSize = 200
		}
	} else {
		if mlp.BatchSize > nSamples {
			log.Printf("Got batchsize larger than sample size.  It is going to be clipped.\n")
			mlp.BatchSize = nSamples
		}
	}
	// # Initialize lists
	activations := []blas64General{X}
	deltas := []blas64General{}
	for _, nFanOut := range layerUnits[1:] {
		activations = append(activations, blas64General{Rows: mlp.BatchSize, Cols: nFanOut, Stride: nFanOut, Data: make([]float64, mlp.BatchSize*nFanOut, mlp.BatchSize*nFanOut)})
		deltas = append(deltas, blas64General{Rows: mlp.BatchSize, Cols: nFanOut, Stride: nFanOut, Data: make([]float64, mlp.BatchSize*nFanOut, mlp.BatchSize*nFanOut)})
	}

	if strings.EqualFold(mlp.Solver, "lbfgs") {
		// # Run the LBFGS solver
		mlp.fitLbfgs(X, y, activations, deltas, mlp.CoefsGrads,
			mlp.InterceptsGrads, layerUnits)
	} else {
		// # Run the Stochastic optimization solver
		mlp.fitStochastic(X, y, activations, deltas, mlp.CoefsGrads,
			mlp.InterceptsGrads, layerUnits, incremental)
	}
}

// Fit compute Coefs and Intercepts
func (mlp *BaseMultilayerPerceptron64) Fit(X, Y Matrix) {
	if xg, ok := X.(RawMatrixer64); ok {
		if yg, ok := Y.(RawMatrixer64); ok {
			mlp.fit(xg.RawMatrix(), yg.RawMatrix(), false)
			return
		}
	}
	var xg, yg General64
	xg.Copy(X)
	yg.Copy(Y)
	mlp.fit(blas64General(xg), blas64General(yg), false)
}

// Predict do forward pass and fills Y (Y must be Mutable)
func (mlp *BaseMultilayerPerceptron64) Predict(X, Y Matrix) {
	if xg, ok := X.(RawMatrixer64); ok {
		if yg, ok := Y.(RawMatrixer64); ok {
			mlp.predict(xg.RawMatrix(), yg.RawMatrix())
			return
		}
	}
	var xg, yg General64
	xg.Copy(X)
	yg.Copy(Y)
	mlp.predict(blas64General(xg), blas64General(yg))

	ymut, ok := Y.(Mutable)
	if !ok {
		log.Panicf("Y bus be mutable")
	}

	for r, rpos := 0, 0; r < yg.Rows; r, rpos = r+1, rpos+yg.Stride {
		for c := 0; c < yg.Cols; c++ {
			ymut.SetAt(r, c, float64(yg.Data[rpos+c]))
		}
	}
}

func (mlp *BaseMultilayerPerceptron64) validateHyperparameters() {
	if mlp.MaxIter <= 0 {
		log.Panicf("maxIter must be > 0, got %d.", mlp.MaxIter)
	}
	if mlp.Alpha < 0.0 {
		log.Panicf("alpha must be >= 0, got %g.", mlp.Alpha)
	}
	if mlp.LearningRateInit <= 0.0 {
		log.Panicf("learningRateInit must be > 0, got %g.", mlp.LearningRateInit)
	}
	if mlp.Momentum > 1 || mlp.Momentum < 0 {
		log.Panicf("momentum must be >= 0 and <= 1, got %g", mlp.Momentum)
	}
	if mlp.ValidationFraction < 0 || mlp.ValidationFraction >= 1 {
		log.Panicf("validationFraction must be >= 0 and < 1, got %g", mlp.ValidationFraction)
	}
	if mlp.Beta1 < 0 || mlp.Beta1 >= 1 {
		log.Panicf("beta_1 must be >= 0 and < 1, got %g", mlp.Beta1)
	}
	if mlp.Beta2 < 0 || mlp.Beta2 >= 1 {
		log.Panicf("beta_2 must be >= 0 and < 1, got %g", mlp.Beta2)
	}
	if mlp.Epsilon <= 0.0 {
		log.Panicf("epsilon must be > 0, got %g.", mlp.Epsilon)
	}
	if mlp.NIterNoChange <= 0 {
		log.Panicf("nIterNoChange must be > 0, got %d.", mlp.NIterNoChange)
	}
	//# raise ValueError if not registered

	supportedActivations := []string{}
	for k := range Activations64 {
		supportedActivations = append(supportedActivations, k)
	}

	if _, ok := Activations64[mlp.Activation]; !ok {
		log.Panicf("The activation \"%s\" is not supported. Supported activations are %s.", mlp.Activation, supportedActivations)
	}
	switch mlp.LearningRate {
	case "constant", "invscaling", "adaptive":
	default:
		log.Panicf("learning rate %s is not supported.", mlp.LearningRate)
	}
	switch mlp.Solver {
	case "sgd", "adam", "lbfgs":
	default:
		log.Panicf("The solver %s is not supported.", mlp.Solver)
	}
}

func (mlp *BaseMultilayerPerceptron64) fitLbfgs(X, y blas64General, activations, deltas, coefGrads []blas64General,
	interceptGrads [][]float64, layerUnits []int) {
	method := &optimize.LBFGS{}
	settings := &optimize.Settings{FuncEvaluations: mlp.MaxIter, GradientThreshold: float64(mlp.Tol)}

	problem := optimize.Problem{
		Func: func(w []float64) float64 {
			for i := range w {
				mlp.packedParameters[i] = float64(w[i])
			}
			return float64(mlp.backprop(X, y, activations, deltas, coefGrads, interceptGrads))

		},
		Grad: func(g, w []float64) []float64 {
			// Grad is called just aveter Func with same w
			if g == nil {
				g = make([]float64, len(w), len(w))
			}
			for i := range w {
				g[i] = float64(mlp.packedGrads[i])
			}
			return g
		},
	}
	w := make([]float64, len(mlp.packedParameters), len(mlp.packedParameters))
	for i := range w {
		w[i] = float64(mlp.packedParameters[i])
	}
	res, err := optimize.Minimize(problem, w, settings, method)
	if err != nil {
		log.Panic(err)
	}
	if res.Status != optimize.GradientThreshold {
		log.Printf("lbfgs optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet.\n", mlp.MaxIter)
	}
}

func (mlp *BaseMultilayerPerceptron64) fitStochastic(X, y blas64General, activations, deltas, coefGrads []blas64General,
	interceptGrads [][]float64, layerUnits []int, incremental bool) {
	if !incremental || mlp.optimizer == Optimizer64(nil) {
		params := mlp.packedParameters
		switch mlp.Solver {
		case "sgd":
			mlp.optimizer = &SGDOptimizer64{
				Params:           params,
				LearningRateInit: mlp.LearningRateInit,
				LearningRate:     mlp.LearningRateInit,
				LRSchedule:       mlp.LearningRate,
				PowerT:           mlp.PowerT,
				Momentum:         mlp.Momentum,
				Nesterov:         mlp.NesterovsMomentum}
		case "adam":
			mlp.optimizer = &AdamOptimizer64{
				Params:           params,
				LearningRateInit: mlp.LearningRateInit,
				LearningRate:     mlp.LearningRateInit,
				Beta1:            mlp.Beta1, Beta2: mlp.Beta2, Epsilon: mlp.Epsilon,
			}
		}
	}
	// # earlyStopping in partialFit doesn"t make sense
	earlyStopping := mlp.EarlyStopping && !incremental
	var XVal, yVal blas64General
	nSamples := X.Rows
	testSize := 0
	if earlyStopping {
		testSize = int(M64.Ceil(mlp.ValidationFraction * float64(nSamples)))
		XVal = blas64General(General64(X).RowSlice(nSamples-testSize, nSamples))
		yVal = blas64General(General64(y).RowSlice(nSamples-testSize, nSamples))
		mlp.bestParameters = make([]float64, len(mlp.packedParameters), len(mlp.packedParameters))
		// if isClassifier(self):
		// 	yVal = self.LabelBinarizer.inverseTransform(yVal)
	}
	batchSize := mlp.BatchSize
	idx := make([]int, nSamples, nSamples)
	for i := range idx {
		idx[i] = i
	}
	type Shuffler interface {
		Shuffle(n int, swap func(i, j int))
	}
	var rndShuffle func(n int, swap func(i, j int))
	if shuffler, ok := mlp.RandomState.(Shuffler); ok {
		rndShuffle = shuffler.Shuffle

	} else {
		rndShuffle = rand.New(mlp.RandomState).Shuffle
	}
	func() {
		if r := recover(); r != nil {
			// ...
			log.Panic(r)
		}
		for it := 0; it < mlp.MaxIter; it++ {
			if mlp.Shuffle {
				rndShuffle(nSamples, indexedXY{idx: sort.IntSlice(idx), X: General64(X), Y: General64(y)}.Swap)
			}
			accumulatedLoss := float64(0.0)
			for batch := [2]int{0, batchSize}; batch[0] < nSamples-testSize; batch = [2]int{batch[1], batch[1] + batchSize} {
				if batch[1] > nSamples-testSize {
					batch[1] = nSamples - testSize
				}
				// activations[0] = X[batchSlice]
				Xbatch := blas64General(General64(X).RowSlice(batch[0], batch[1]))
				Ybatch := blas64General(General64(y).RowSlice(batch[0], batch[1]))

				activations[0] = Xbatch
				for _, a := range activations {
					a.Rows = Xbatch.Rows
				}

				//X, y blas64General, activations, deltas, coefGrads []blas64General, interceptGrads
				batchLoss := mlp.backprop(Xbatch, Ybatch, activations, deltas, mlp.CoefsGrads, mlp.InterceptsGrads)
				accumulatedLoss += batchLoss * float64(batch[1]-batch[0])

				//# update weights
				mlp.optimizer.updateParams(mlp.packedGrads)
			}
			mlp.NIter++
			mlp.Loss = accumulatedLoss / float64(nSamples)

			mlp.t += nSamples
			mlp.LossCurve = append(mlp.LossCurve, mlp.Loss)
			if mlp.Verbose {
				fmt.Printf("Iteration %d, loss = %.8f\n", mlp.NIter, mlp.Loss)
			}
			// # update noImprovementCount based on training loss or
			// # validation score according to earlyStopping
			mlp.updateNoImprovementCount(earlyStopping, XVal, yVal)

			// # for learning rate that needs to be updated at iteration end
			mlp.optimizer.iterationEnds(float64(mlp.t))

			if mlp.NoImprovementCount > mlp.NIterNoChange {
				// # not better than last `nIterNoChange` iterations by tol
				// # stop or decrease learning rate
				var msg string
				if earlyStopping {
					msg = fmt.Sprintf("Validation score did not improve more than tol=%f for %d consecutive epochs.", mlp.Tol, mlp.NIterNoChange)
				} else {
					msg = fmt.Sprintf("Training loss did not improve more than tol=%f for %d consecutive epochs.", mlp.Tol, mlp.NIterNoChange)
				}
				isStopping := mlp.optimizer.triggerStopping(msg, mlp.Verbose)
				if isStopping {
					break
				}
				mlp.NoImprovementCount = 0
			}

			if incremental {
				break
			}
			if mlp.NIter == mlp.MaxIter && mlp.MaxIter > 1 {
				log.Printf("Stochastic Optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet.\n", mlp.MaxIter)
			}
		}
	}()
	if earlyStopping {
		// # restore best weights
		copy(mlp.packedParameters, mlp.bestParameters)
	}
	if mlp.Shuffle {
		sort.Sort(indexedXY{idx: sort.IntSlice(idx), X: General64(X), Y: General64(y)})
	}
}

func (mlp *BaseMultilayerPerceptron64) updateNoImprovementCount(earlyStopping bool, XVal, yVal blas64General) {

	if earlyStopping {
		//# compute validation score, use that for stopping
		lastValidScore := mlp.score(XVal, yVal)

		mlp.ValidationScores = append(mlp.ValidationScores, lastValidScore)

		if mlp.Verbose {
			fmt.Printf("Validation score: %g\n", lastValidScore)
		}
		// # update best parameters
		// # use validationScores_, not lossCurve_
		// # let's hope no-one overloads .score with mse

		if lastValidScore < (mlp.BestValidationScore + mlp.Tol) {
			mlp.NoImprovementCount++
		} else {
			mlp.NoImprovementCount = 0
		}
		if lastValidScore > mlp.BestValidationScore {
			mlp.BestValidationScore = lastValidScore
			copy(mlp.bestParameters, mlp.packedParameters)
		}
	} else {
		lastLoss := mlp.LossCurve[len(mlp.LossCurve)-1]
		if lastLoss > mlp.BestLoss-mlp.Tol {
			mlp.NoImprovementCount++
		} else {
			mlp.NoImprovementCount = 0
		}
		if lastLoss < mlp.BestLoss {
			mlp.BestLoss = lastLoss
		}
	}
}
func (mlp *BaseMultilayerPerceptron64) predict(X, Y blas64General) {
	_, nFeatures := X.Rows, X.Cols

	layerUnits := append([]int{nFeatures}, mlp.HiddenLayerSizes...)
	layerUnits = append(layerUnits, mlp.NOutputs)
	// # Initialize layers
	activations := []blas64General{X}
	for i, nFanOut := range layerUnits[1:] {
		var activation blas64General
		if i == len(layerUnits)-2 {
			activation = Y
		} else {
			activation = blas64General{Rows: X.Rows, Cols: nFanOut, Stride: nFanOut, Data: make([]float64, X.Rows*nFanOut, X.Rows*nFanOut)}
		}
		activations = append(activations, activation)
	}
	// # forward propagate
	mlp.forwardPass(activations)
	if strings.EqualFold(mlp.OutActivation, "logistic") || strings.EqualFold(mlp.OutActivation, "softmax") {
		toLogits64(Y)
	}
}

func (mlp *BaseMultilayerPerceptron64) score(X, Y blas64General) float64 {
	H := blas64General{Rows: Y.Rows, Cols: Y.Cols, Stride: Y.Stride, Data: make([]float64, len(Y.Data), len(Y.Data))}
	mlp.predict(X, H)
	if mlp.LossFuncName != "square_loss" {
		// accuracy
		return accuracyScore64(Y, H)
	}
	// R2Score
	return r2Score64(Y, H)

}

func (mlp *BaseMultilayerPerceptron64) validateInput(X, y blas64General, incremental bool) (Xout, youy blas64General) {
	/*
	   X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
	                     multi_output=True)
	    if y.ndim == 2 and y.shape[1] == 1:
	        y = column_or_1d(y, warn=True)

	    if not incremental:
	        self._label_binarizer = LabelBinarizer()
	        self._label_binarizer.fit(y)
	        self.classes_ = self._label_binarizer.classes_
	    elif self.warm_start:
	        classes = unique_labels(y)
	        if set(classes) != set(self.classes_):
	            raise ValueError("warm_start can only be used where `y` has "
	                             "the same classes as in the previous "
	                             "call to fit. Previously got %s, `y` has %s" %
	                             (self.classes_, classes))
	    else:
	        classes = unique_labels(y)
	        if len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
	            raise ValueError("`y` has classes not in `self.classes_`."
	                             " `self.classes_` has %s. 'y' has %s." %
	                             (self.classes_, classes))

	    y = self._label_binarizer.transform(y)
	    return X, y
	*/
	return X, y
}

// SGDOptimizer64 is the stochastic gradient descent optimizer
type SGDOptimizer64 struct {
	Params           []float64
	LearningRateInit float64
	LearningRate     float64
	PowerT           float64
	LRSchedule       string
	Momentum         float64
	Nesterov         bool
	velocities       []float64
}

func (opt *SGDOptimizer64) iterationEnds(timeStep float64) {
	if strings.EqualFold(opt.LRSchedule, "invscaling") {
		opt.LearningRate = opt.LearningRateInit / M64.Pow(timeStep+1, opt.PowerT)
	}

}
func (opt *SGDOptimizer64) triggerStopping(msg string, verbose bool) bool {
	if !strings.EqualFold(opt.LRSchedule, "adaptive") {
		if verbose {
			fmt.Println(msg + " Stopping.")
		}
		return true
	}
	if opt.LearningRate <= 1e-6 {
		if verbose {
			fmt.Println(msg + " Learning rate too small. Stopping.")
		}
		return true
	}
	opt.LearningRate /= 5.
	if verbose {
		fmt.Println(msg+" Setting learning rate to %f", opt.LearningRate)
	}
	return false
}
func (opt *SGDOptimizer64) updateParams(grads []float64) {
	if opt.velocities == nil {
		opt.velocities = make([]float64, len(grads), len(grads))
	}
	for i := range grads {
		update := opt.Momentum*opt.velocities[i] - opt.LearningRate*grads[i]
		velocity := update
		opt.velocities[i] = velocity
		if opt.Nesterov {
			opt.Params[i] += opt.Momentum*velocity - opt.LearningRate*grads[i]
		} else {
			opt.Params[i] += update
		}
	}

}

// AdamOptimizer64 is the stochastic adam optimizer
type AdamOptimizer64 struct {
	Params                []float64
	LearningRateInit      float64
	LearningRate          float64
	Beta1, Beta2, Epsilon float64
	t                     float64
	ms, vs                []float64
}

func (opt *AdamOptimizer64) iterationEnds(timeStep float64)                {}
func (opt *AdamOptimizer64) triggerStopping(msg string, verbose bool) bool { return true }
func (opt *AdamOptimizer64) updateParams(grads []float64) {
	if opt.t == 0 {
		opt.ms = make([]float64, len(grads), len(grads))
		opt.vs = make([]float64, len(grads), len(grads))
	}
	opt.t++
	for i, grad := range grads {
		opt.ms[i] = opt.Beta1*opt.ms[i] + (1-opt.Beta1)*grad
		opt.vs[i] = opt.Beta2*opt.vs[i] + (1-opt.Beta2)*grad*grad
		opt.LearningRate = opt.LearningRateInit * M64.Sqrt(1-M64.Pow(opt.Beta2, opt.t)) / (1. - M64.Pow(opt.Beta1, opt.t))
		update := -opt.LearningRate * opt.ms[i] / (M64.Sqrt(opt.vs[i]) + opt.Epsilon)
		opt.Params[i] += update
	}
}

func toLogits64(ym blas64General) {
	for i, ypos := 0, 0; i < ym.Rows; i, ypos = i+1, ypos+ym.Stride {
		if ym.Cols == 1 {
			v := ym.Data[ypos]
			if v > .5 {
				v = 1
			} else {
				v = 0
			}
			ym.Data[ypos] = v
		} else {
			M := MaxIdx64(ym.Data[ypos : ypos+ym.Cols])
			for c := 0; c < ym.Cols; c++ {
				v := float64(0)
				if c == M {
					v = 1.
				}
				ym.Data[ypos+c] = v
			}
		}
	}
}

func r2Score64(yTrue, yPred blas64General) float64 {
	var r2acc float64
	for c := 0; c < yTrue.Cols; c++ {
		// yTrueAvg
		var yTrueAvg, yNum, yDen float64
		for r, ypos := 0, 0; r < yTrue.Rows; r, ypos = r+1, ypos+yTrue.Stride {
			yTrueAvg += yTrue.Data[ypos+c]

		}
		yTrueAvg /= float64(yTrue.Rows)
		// yDen = YTrue-YTrueAvg

		for r, ypos := 0, 0; r < yTrue.Rows; r, ypos = r+1, ypos+yTrue.Stride {
			t := yPred.Data[ypos+c] - yTrue.Data[ypos+c]
			yNum += t * t
			t = yTrue.Data[ypos+c] - yTrueAvg
			yDen += t * t
		}
		if yDen == 0 {
			panic("yDen=0")
		}
		r2 := 1 - yNum/yDen
		r2acc += r2
	}
	return r2acc / float64(yTrue.Cols)
}

func accuracyScore64(Y, H blas64General) float64 {
	N := float64(0)
	for i, hpos, ypos := 0, 0, 0; i < Y.Rows; i, hpos, ypos = i+1, hpos+H.Stride, ypos+Y.Stride {
		rowEq := true
		for c := 0; c < Y.Cols; c++ {
			rowEq = rowEq && H.Data[hpos+c] == Y.Data[ypos+c]
		}
		if rowEq {
			N++
		}
	}
	return float64(N) / float64(Y.Rows)

}
