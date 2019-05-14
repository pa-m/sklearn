package neuralnetwork

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/blas/blas32"

	"github.com/pa-m/sklearn/base"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// BaseMultilayerPerceptron32 closely matches sklearn/neural_network/multilayer_perceptron.py
type BaseMultilayerPerceptron32 struct {
	Activation         string  `json:"activation"`
	Solver             string  `json:"solver"`
	Alpha              float32 `json:"alpha"`
	WeightDecay        float32 `json:"weight_decay"`
	BatchSize          int     `json:"batch_size"`
	BatchNormalize     bool
	LearningRate       string           `json:"learning_rate"`
	LearningRateInit   float32          `json:"learning_rate_init"`
	PowerT             float32          `json:"power_t"`
	MaxIter            int              `json:"max_iter"`
	LossFuncName       string           `json:"loss_func_name"`
	HiddenLayerSizes   []int            `json:"hidden_layer_sizes"`
	Shuffle            bool             `json:"shuffle"`
	RandomState        base.RandomState `json:"random_state"`
	Tol                float32          `json:"tol"`
	Verbose            bool             `json:"verbose"`
	WarmStart          bool             `json:"warm_start"`
	Momentum           float32          `json:"momentum"`
	NesterovsMomentum  bool             `json:"nesterovs_momentum"`
	EarlyStopping      bool             `json:"early_stopping"`
	ValidationFraction float32          `json:"validation_fraction"`
	Beta1              float32          `json:"beta_1"`
	Beta2              float32          `json:"beta_2"`
	Epsilon            float32          `json:"epsilon"`
	NIterNoChange      int              `json:"n_iter_no_change"`

	// Outputs
	NLayers       int
	NIter         int
	NOutputs      int
	Intercepts    [][]float32     `json:"intercepts_"`
	Coefs         []blas32General `json:"coefs_"`
	OutActivation string          `json:"out_activation_"`
	Loss          float32

	// internal
	t                   int
	LossCurve           []float32
	ValidationScores    []float32
	BestValidationScore float32
	BestLoss            float32
	NoImprovementCount  int
	optimizer           Optimizer32
	packedParameters    []float32
	packedGrads         []float32 // packedGrads allow tests to check gradients
	bestParameters      []float32
	batchNorm           [][]float32
	lb                  *LabelBinarizer32
	// beforeMinimize allow test to set weights
	beforeMinimize func(optimize.Problem, []float64)
}

// Activations32 is a map containing the inplace_activation functions
var Activations32 = map[string]func(z blas32General){
	"identity": func(z blas32General) {},
	"logistic": func(z blas32General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] = 1 / (1 + M32.Exp(-z.Data[zpos+col]))
			}
		}
	},
	"tanh": func(z blas32General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] = M32.Tanh(-z.Data[zpos+col])
			}
		}
	},
	"relu": func(z blas32General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				if z.Data[zpos+col] < 0 {
					z.Data[zpos+col] = 0
				}
			}
		}
	},
	"softmax": func(z blas32General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			sum := float32(0)
			for col := 0; col < z.Cols; col++ {

				z.Data[zpos+col] = M32.Exp(z.Data[zpos+col])
				sum += z.Data[zpos+col]
			}
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] /= sum
			}
		}
	},
}

// Derivatives32 is a map of functions which multiply deltas with derivative of activation function
var Derivatives32 = map[string]func(Z, deltas blas32General){
	"identity": func(Z, deltas blas32General) {
	},
	"logistic": func(Z, deltas blas32General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				z := Z.Data[zpos+col]
				deltas.Data[dpos+col] *= z * (1 - z)
			}
		}
	},
	"tanh": func(Z, deltas blas32General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				z := Z.Data[zpos+col]
				deltas.Data[dpos+col] *= 1 - z*z
			}
		}
	},
	"relu": func(Z, deltas blas32General) {
		for row, zpos, dpos := 0, 0, 0; row < Z.Rows; row, zpos, dpos = row+1, zpos+Z.Stride, dpos+deltas.Stride {
			for col := 0; col < Z.Cols; col++ {
				if Z.Data[zpos+col] == 0 {
					deltas.Data[dpos+col] = 0
				}
			}
		}
	},
}

// LossFunctions32 is a map for loss functions
var LossFunctions32 = map[string]func(y, h blas32General) float32{
	"square_loss": func(y, h blas32General) float32 {
		sum := float32(0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				e := h.Data[hpos+col] - y.Data[ypos+col]
				sum += e * e
			}
		}
		return sum / 2 / float32(h.Rows)
	},
	"log_loss": func(y, h blas32General) float32 {
		sum := float32(0)
		hmin, hmax := M32.Nextafter(0, 1), M32.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				if y.Data[ypos+col] != 0 {
					sum += -y.Data[ypos+col] * M32.Log(hval)
				}
			}
		}
		return sum / float32(h.Rows)
	},
	"binary_log_loss": func(y, h blas32General) float32 {
		sum := float32(0)
		hmin, hmax := M32.Nextafter(0, 1), M32.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				sum += -y.Data[ypos+col]*M32.Log(hval) - (1-y.Data[ypos+col])*M32.Log1p(-hval)
			}
		}
		return sum / float32(h.Rows)
	},
}

// Optimizer32 is an interface for stochastic optimizers
type Optimizer32 interface {
	iterationEnds(timeStep float32)
	triggerStopping(msg string, verbose bool) bool
	updateParams(grads []float32)
}

func addIntercepts32(a blas32General, b []float32) {
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			a.Data[apos+c] += b[c]
		}
	}
}

func matRowMean32(a blas32General, b []float32) {
	for c := 0; c < a.Cols; c++ {
		b[c] = 0
	}
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			b[c] += a.Data[apos+c]
		}
	}
	for c := 0; c < a.Cols; c++ {
		b[c] /= float32(a.Rows)
	}
}

// NewBaseMultilayerPerceptron32 returns a BaseMultilayerPerceptron32 with defaults
func NewBaseMultilayerPerceptron32() *BaseMultilayerPerceptron32 {
	return &BaseMultilayerPerceptron32{

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
		//RandomState        base.Source,
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
//        activations : []blas32General, length = nLayers - 1
func (mlp *BaseMultilayerPerceptron32) forwardPass(activations []blas32General) {
	hiddenActivation := Activations32[mlp.Activation]
	var i int
	for i = 0; i < mlp.NLayers-1; i++ {
		gemm32(blas.NoTrans, blas.NoTrans, 1, activations[i], mlp.Coefs[i], 0, activations[i+1])
		addIntercepts32(activations[i+1], mlp.Intercepts[i])
		// For the hidden layers
		if (i + 1) != (mlp.NLayers - 1) {
			hiddenActivation(activations[i+1])
		}
	}
	i = mlp.NLayers - 2
	// # For the last layer
	outputActivation := Activations32[mlp.OutActivation]
	outputActivation(activations[i+1])
}

// batchNormalize computes norms of activations and divides activations
func (mlp *BaseMultilayerPerceptron32) batchNormalize(activations []blas32General) {
	for i := 0; i < mlp.NLayers-2; i++ {
		activation := activations[i+1]
		batchNorm := mlp.batchNorm[i]
		for o := 0; o < activation.Cols; o++ {
			M := float32(0)
			// compute max for layer i, output o
			for r, rpos := 0, 0; r < activation.Rows; r, rpos = r+1, rpos+activation.Stride {
				a := M32.Abs(activation.Data[rpos+o])
				if M < a {
					M = a
				}
			}
			// divide activation by max
			if M > 0 {
				for r, rpos := 0, 0; r < activation.Rows; r, rpos = r+1, rpos+activation.Stride {
					activation.Data[rpos+o] /= M
				}
			}
			batchNorm[o] = M
		}
	}
}

// batchNormalizeDeltas divides deltas by batchNorm
func (mlp *BaseMultilayerPerceptron32) batchNormalizeDeltas(deltas blas32General, batchNorm []float32) {
	for r, rpos := 0, 0; r < deltas.Rows; r, rpos = r+1, rpos+deltas.Stride {
		for o := 0; o < deltas.Cols; o++ {
			deltas.Data[rpos+o] /= batchNorm[o]
		}
	}
}

func (mlp *BaseMultilayerPerceptron32) sumCoefSquares() float32 {
	s := float32(0)
	for _, c := range mlp.Coefs {
		for _, co := range c.Data {
			s += co * co
		}
	}
	return s
}

// computeLossGrad Compute the gradient of loss with respect to coefs and intercept for specified layer.
// This function does backpropagation for the specified one layer.
func (mlp *BaseMultilayerPerceptron32) computeLossGrad(layer, NSamples int, activations []blas32General, deltas []blas32General, coefGrads []blas32General, interceptGrads [][]float32) {
	// coefGrads[layer] = safeSparseDot(activations[layer].T, deltas[layer])
	// coefGrads[layer] += (self.alpha * self.coefs_[layer])
	// coefGrads[layer] /= nSamples
	gemm32(blas.Trans, blas.NoTrans, 1/float32(NSamples), activations[layer], deltas[layer], 0, coefGrads[layer])
	axpy32(len(coefGrads[layer].Data), mlp.Alpha/float32(NSamples), mlp.Coefs[layer].Data, coefGrads[layer].Data)
	// interceptGrads[layer] = np.mean(deltas[layer], 0)
	matRowMean32(deltas[layer], interceptGrads[layer])
}

// backprop Compute the MLP loss function and its corresponding derivatives with respect to each parameter: weights and bias vectors.
// X : blas32General shape (nSamples, nFeatures)
// Y : blas32General shape (nSamples, nOutputs)
// activations : []blas32General, length=NLayers-1
// deltas : []blas32General, length=NLayers-1
// coefGrads : []blas32General, length=NLayers-1
// interceptGrads : [][]float32, length=NLayers-1

func (mlp *BaseMultilayerPerceptron32) backprop(X, y blas32General, activations, deltas, coefGrads []blas32General, interceptGrads [][]float32) float32 {
	nSamples := X.Rows
	if mlp.WeightDecay > 0 {
		for iw := range mlp.packedParameters {
			mlp.packedParameters[iw] *= (1 - mlp.WeightDecay)
		}
	}
	mlp.forwardPass(activations)
	if mlp.BatchNormalize {
		// compute norm of activations for non-terminal layers
		mlp.batchNormalize(activations)
	}

	//# Get loss
	lossFuncName := mlp.LossFuncName
	if strings.EqualFold(lossFuncName, "log_loss") && strings.EqualFold(mlp.OutActivation, "logistic") {
		lossFuncName = "binary_log_loss"
	}
	// y may have less rows than activations il last batch
	loss := LossFunctions32[lossFuncName](y, activations[len(activations)-1])
	// # Add L2 regularization term to loss
	loss += (0.5 * mlp.Alpha) * mlp.sumCoefSquares() / float32(nSamples)

	//# Backward propagate
	last := mlp.NLayers - 2

	// # The calculation of delta[last] here works with following
	// # combinations of output activation and loss function:
	// # sigmoid and binary cross entropy, softmax and categorical cross
	// # entropy, and identity with squared loss
	//deltas[last] = activations[len(activations)-1] - y

	// y may have less rows than activations il last batch
	{
		H := activations[len(activations)-1]
		D := deltas[last]
		for r, pos := 0, 0; r < y.Rows; r, pos = r+1, pos+y.Stride {
			for o, posc := 0, pos; o < y.Cols; o, posc = o+1, posc+1 {
				D.Data[posc] = H.Data[posc] - y.Data[posc]
			}
		}
	}

	//# Compute gradient for the last layer
	mlp.computeLossGrad(
		last, nSamples, activations, deltas, coefGrads, interceptGrads)

	//# Iterate over the hidden layers
	for i := mlp.NLayers - 2; i >= 1; i-- {
		//deltas[i - 1] = safeSparseDot(deltas[i], self.coefs_[i].T)
		gemm32(blas.NoTrans, blas.Trans, 1, deltas[i], mlp.Coefs[i], 0, deltas[i-1])

		inplaceDerivative := Derivatives32[mlp.Activation]
		// inplaceDerivative multiplies deltas[i-1] by activation derivative
		inplaceDerivative(activations[i], deltas[i-1])
		if mlp.BatchNormalize {
			// divide deltas by batchNorm
			mlp.batchNormalizeDeltas(deltas[i-1], mlp.batchNorm[i-1])
		}

		mlp.computeLossGrad(
			i-1, nSamples, activations, deltas, coefGrads,
			interceptGrads)

	}
	return loss
}

func (mlp *BaseMultilayerPerceptron32) initialize(yCols int, layerUnits []int, isClassifier, isMultiClass bool) {
	// # set all attributes, allocate weights etc for first call
	// # Initialize parameters
	mlp.NIter = 0
	mlp.t = 0
	mlp.NOutputs = yCols

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
	mlp.Coefs = make([]blas32General, mlp.NLayers-1)
	mlp.Intercepts = make([][]float32, mlp.NLayers-1)
	off := 0
	for i := 0; i < mlp.NLayers-1; i++ {
		off += (1 + layerUnits[i]) * layerUnits[i+1]
	}
	mem := make([]float32, off)
	mlp.packedParameters = mem[0:off]
	if mlp.BatchNormalize {
		// allocate batchNorm for non-terminal layers
		mlp.batchNorm = make([][]float32, mlp.NLayers-2)
	}

	off = 0
	if mlp.RandomState == (base.RandomState)(nil) {
		mlp.RandomState = base.NewLockedSource(uint64(time.Now().UnixNano()))
	}
	type Float32er interface {
		Float32() float32
	}
	var rndFloat32 func() float32
	if float32er, ok := mlp.RandomState.(Float32er); ok {
		rndFloat32 = float32er.Float32
	} else {
		rndFloat32 = rand.New(mlp.RandomState).Float32
	}
	for i := 0; i < mlp.NLayers-1; i++ {
		prevOff := off
		mlp.Intercepts[i] = mem[off : off+layerUnits[i+1]]
		off += layerUnits[i+1]
		mlp.Coefs[i] = blas32General{Rows: layerUnits[i], Cols: layerUnits[i+1], Stride: layerUnits[i+1], Data: mem[off : off+layerUnits[i]*layerUnits[i+1]]}
		off += layerUnits[i] * layerUnits[i+1]
		// # Use the initialization method recommended by
		// # Glorot et al.
		factor := float32(6.)
		fanIn, fanOut := layerUnits[i], layerUnits[i+1]
		if strings.EqualFold(mlp.Activation, "logistic") {
			factor = 2.
		}

		initBound := M32.Sqrt(factor / float32(fanIn+fanOut))
		for pos := prevOff; pos < off; pos++ {
			mem[pos] = rndFloat32() * initBound
		}
		if mlp.BatchNormalize && i < mlp.NLayers-2 {
			mlp.batchNorm[i] = make([]float32, layerUnits[i+1])
		}
	}

	mlp.BestLoss = M32.Inf(1)
}

func (mlp *BaseMultilayerPerceptron32) fit(X, y blas32General, incremental bool) {
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
		mlp.initialize(y.Cols, layerUnits, isClassifier, isMulticlass)
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
	activations := make([]blas32.General, 1, len(layerUnits))
	activations[0] = X
	deltas := make([]blas32.General, 0, len(layerUnits)-1)
	// compute size of activations and deltas
	off := 0
	for _, nFanOut := range layerUnits[1:] {
		size := mlp.BatchSize * nFanOut
		off += size + size
	}
	mem := make([]float32, off)
	off = 0
	for _, nFanOut := range layerUnits[1:] {
		size := mlp.BatchSize * nFanOut
		activations = append(activations, blas32General{Rows: mlp.BatchSize, Cols: nFanOut, Stride: nFanOut, Data: mem[off : off+size]})
		off += size
		deltas = append(deltas, blas32General{Rows: mlp.BatchSize, Cols: nFanOut, Stride: nFanOut, Data: mem[off : off+size]})
		off += size
	}

	off = len(mlp.packedParameters)
	packedGrads := make([]float32, off)
	CoefsGrads := make([]blas32General, mlp.NLayers-1)
	InterceptsGrads := make([][]float32, mlp.NLayers-1)
	off = 0
	for i := 0; i < mlp.NLayers-1; i++ {
		InterceptsGrads[i] = packedGrads[off : off+layerUnits[i+1]]
		off += layerUnits[i+1]
		CoefsGrads[i] = blas32General{Rows: layerUnits[i], Cols: layerUnits[i+1], Stride: layerUnits[i+1], Data: packedGrads[off : off+layerUnits[i]*layerUnits[i+1]]}
		off += layerUnits[i] * layerUnits[i+1]
	}

	if strings.EqualFold(mlp.Solver, "lbfgs") {
		// # Run the LBFGS solver
		mlp.fitLbfgs(X, y, activations, deltas, CoefsGrads,
			InterceptsGrads, packedGrads, layerUnits)
	} else {
		// # Run the Stochastic optimization solver
		mlp.fitStochastic(X, y, activations, deltas, CoefsGrads,
			InterceptsGrads, packedGrads, layerUnits, incremental)
	}
	mlp.packedGrads = packedGrads
}

// IsClassifier return true if LossFuncName is not square_loss
func (mlp *BaseMultilayerPerceptron32) IsClassifier() bool {
	return mlp.LossFuncName != "square_loss"
}

// Fit compute Coefs and Intercepts
func (mlp *BaseMultilayerPerceptron32) Fit(X, Y Matrix) {
	var xb, yb blas32.General
	if xg, ok := X.(RawMatrixer32); ok && !mlp.Shuffle {
		if yg, ok := Y.(RawMatrixer32); ok {
			xb, yb = xg.RawMatrix(), yg.RawMatrix()
		}
	} else {
		var tmp General32
		tmp = General32(xb)
		tmp.Copy(X)
		xb = tmp.RawMatrix()
		tmp = General32(yb)
		tmp.Copy(Y)
		yb = tmp.RawMatrix()
	}
	if mlp.IsClassifier() && !isBinarized32(yb) {
		mlp.lb = NewLabelBinarizer32(0, 1)
		xbin, ybin := mlp.lb.FitTransform(General32(xb), General32(yb))
		xb, yb = blas32.General(xbin), blas32.General(ybin)
	}
	mlp.fit(xb, yb, false)
}

// GetNOutputs returns output columns number for Y to pass to predict
func (mlp *BaseMultilayerPerceptron32) GetNOutputs() int {
	if mlp.lb != nil {
		return len(mlp.lb.Classes)
	}
	return mlp.NOutputs
}

// Predict do forward pass and fills Y (Y must be Mutable)
func (mlp *BaseMultilayerPerceptron32) Predict(X mat.Matrix, Y Mutable) {
	var xb, yb General32
	if xg, ok := X.(RawMatrixer32); ok {
		if yg, ok := Y.(RawMatrixer32); ok {
			xb, yb = General32(xg.RawMatrix()), General32(yg.RawMatrix())
		}
	} else {
		xb.Copy(X)
		yb.Copy(Y)
	}
	mlp.predict(xb.RawMatrix(), yb.RawMatrix())

	FromDense32(Y, yb)
}

func (mlp *BaseMultilayerPerceptron32) validateHyperparameters() {
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
	for k := range Activations32 {
		supportedActivations = append(supportedActivations, k)
	}

	if _, ok := Activations32[mlp.Activation]; !ok {
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

func (mlp *BaseMultilayerPerceptron32) fitLbfgs(X, y blas32General, activations, deltas, coefGrads []blas32General,
	interceptGrads [][]float32, packedGrads []float32, layerUnits []int) {
	method := &optimize.LBFGS{}
	settings := &optimize.Settings{
		FuncEvaluations: mlp.MaxIter,
		Converger: &optimize.FunctionConverge{
			Relative:   float64(mlp.Tol),
			Iterations: mlp.NIterNoChange,
		},
		Concurrent: runtime.GOMAXPROCS(0),
	}

	var mu sync.Mutex // sync access to mlp.Loss on LossCurve
	problem := optimize.Problem{
		Func: func(w []float64) float64 {
			for i := range w {
				mlp.packedParameters[i] = float32(w[i])
			}
			loss := float64(mlp.backprop(X, y, activations, deltas, coefGrads, interceptGrads))
			mu.Lock()
			mlp.Loss = float32(loss)
			mlp.LossCurve = append(mlp.LossCurve, mlp.Loss)
			if mlp.BestLoss > mlp.Loss {
				mlp.BestLoss = mlp.Loss
			}
			mu.Unlock()
			return loss
		},
		Grad: func(g, w []float64) {
			// Grad is called just after Func with same w
			if g == nil { // g is nil at first call
				g = make([]float64, len(w))
			}
			for i := range w {
				g[i] = float64(packedGrads[i])
			}
		},
	}
	w := make([]float64, len(mlp.packedParameters))
	for i := range w {
		w[i] = float64(mlp.packedParameters[i])
	}
	if mlp.beforeMinimize != nil {
		mlp.beforeMinimize(problem, w)
	}
	res, err := optimize.Minimize(problem, w, settings, method)
	if err != nil {
		log.Panic(err)
	}
	if res.Status != optimize.GradientThreshold && res.Status != optimize.FunctionConvergence {
		log.Printf("lbfgs optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet.\n", mlp.MaxIter)
	}
}

func (mlp *BaseMultilayerPerceptron32) fitStochastic(X, y blas32General, activations, deltas, coefGrads []blas32General,
	interceptGrads [][]float32, packedGrads []float32, layerUnits []int, incremental bool) {
	if !incremental || mlp.optimizer == Optimizer32(nil) {
		params := mlp.packedParameters
		switch mlp.Solver {
		case "sgd":
			mlp.optimizer = &SGDOptimizer32{
				Params:           params,
				LearningRateInit: mlp.LearningRateInit,
				LearningRate:     mlp.LearningRateInit,
				LRSchedule:       mlp.LearningRate,
				PowerT:           mlp.PowerT,
				Momentum:         mlp.Momentum,
				Nesterov:         mlp.NesterovsMomentum}
		case "adam":
			mlp.optimizer = &AdamOptimizer32{
				Params:           params,
				LearningRateInit: mlp.LearningRateInit,
				LearningRate:     mlp.LearningRateInit,
				Beta1:            mlp.Beta1, Beta2: mlp.Beta2, Epsilon: mlp.Epsilon,
			}
		}
	}
	// # earlyStopping in partialFit doesn"t make sense
	earlyStopping := mlp.EarlyStopping && !incremental
	var XVal, yVal blas32General
	nSamples := X.Rows
	testSize := 0
	if earlyStopping {
		testSize = int(M32.Ceil(mlp.ValidationFraction * float32(nSamples)))
		XVal = blas32General(General32(X).RowSlice(nSamples-testSize, nSamples))
		yVal = blas32General(General32(y).RowSlice(nSamples-testSize, nSamples))
		mlp.bestParameters = make([]float32, len(mlp.packedParameters))
		// if isClassifier(self):
		// 	yVal = self.LabelBinarizer32.inverseTransform(yVal)
	}
	batchSize := mlp.BatchSize
	idx := make([]int, nSamples)
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
				rndShuffle(nSamples, indexedXY{idx: sort.IntSlice(idx), X: general32FastSwap(X), Y: general32FastSwap(y)}.Swap)
			}
			accumulatedLoss := float32(0.0)
			for batch := [2]int{0, batchSize}; batch[0] < nSamples-testSize; batch = [2]int{batch[1], batch[1] + batchSize} {
				if batch[1] > nSamples-testSize {
					batch[1] = nSamples - testSize
				}
				// activations[0] = X[batchSlice]
				Xbatch := blas32General(General32(X).RowSlice(batch[0], batch[1]))
				Ybatch := blas32General(General32(y).RowSlice(batch[0], batch[1]))

				activations[0] = Xbatch
				for _, a := range activations {
					a.Rows = Xbatch.Rows
				}

				//X, y blas32General, activations, deltas, coefGrads []blas32General, interceptGrads
				batchLoss := mlp.backprop(Xbatch, Ybatch, activations, deltas, coefGrads, interceptGrads)
				accumulatedLoss += batchLoss * float32(batch[1]-batch[0])

				//# update weights
				mlp.optimizer.updateParams(packedGrads)
			}
			mlp.NIter++
			mlp.Loss = accumulatedLoss / float32(nSamples)

			mlp.t += nSamples
			mlp.LossCurve = append(mlp.LossCurve, mlp.Loss)
			if mlp.Verbose {
				fmt.Printf("Iteration %d, loss = %.8f\n", mlp.NIter, mlp.Loss)
			}
			// # update noImprovementCount based on training loss or
			// # validation score according to earlyStopping
			mlp.updateNoImprovementCount(earlyStopping, XVal, yVal)

			// # for learning rate that needs to be updated at iteration end
			mlp.optimizer.iterationEnds(float32(mlp.t))

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
		sort.Sort(indexedXY{idx: sort.IntSlice(idx), X: general32FastSwap(X), Y: general32FastSwap(y)})
	}
}

func (mlp *BaseMultilayerPerceptron32) updateNoImprovementCount(earlyStopping bool, XVal, yVal blas32General) {

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
	}
	lastLoss := mlp.LossCurve[len(mlp.LossCurve)-1]
	if !earlyStopping {
		if lastLoss > mlp.BestLoss-mlp.Tol {
			mlp.NoImprovementCount++
		} else {
			mlp.NoImprovementCount = 0
		}
	}
	if lastLoss < mlp.BestLoss {
		mlp.BestLoss = lastLoss
	}
}

func (mlp *BaseMultilayerPerceptron32) predictProbas(X, Y blas32General) {
	_, nFeatures := X.Rows, X.Cols

	layerUnits := append([]int{nFeatures}, mlp.HiddenLayerSizes...)
	layerUnits = append(layerUnits, mlp.NOutputs)
	// # Initialize layers
	activations := []blas32General{X}
	for i, nFanOut := range layerUnits[1:] {
		var activation blas32General
		if i == len(layerUnits)-2 {
			activation = Y
		} else {
			activation = blas32General{Rows: X.Rows, Cols: nFanOut, Stride: nFanOut, Data: make([]float32, X.Rows*nFanOut)}
		}
		activations = append(activations, activation)
	}
	// # forward propagate
	mlp.forwardPass(activations)
}

func (mlp *BaseMultilayerPerceptron32) predict(X, Y blas32General) {
	var ybin General32
	if mlp.lb == nil {
		ybin = General32(Y)
	} else {
		_, ybin = mlp.lb.Transform(General32(X), General32(Y))
	}
	mlp.predictProbas(X, ybin.RawMatrix())
	if mlp.lb != nil {
		_, Yclasses := mlp.lb.InverseTransform(General32(X), ybin)
		var tmp = General32(Y)
		tmp.Copy(Yclasses)
		Y = tmp.RawMatrix()
	} else if mlp.IsClassifier() {
		toLogits32(Y)
	}
}

func (mlp *BaseMultilayerPerceptron32) score(X, Y blas32General) float32 {
	H := blas32General{Rows: Y.Rows, Cols: Y.Cols, Stride: Y.Stride, Data: make([]float32, len(Y.Data))}
	mlp.predict(X, H)
	if mlp.LossFuncName != "square_loss" {
		// accuracy
		return accuracyScore32(Y, H)
	}
	// R2Score
	return r2Score32(Y, H)
}

func (mlp *BaseMultilayerPerceptron32) validateInput(X, y blas32General, incremental bool) (Xout, youy blas32General) {
	/*
	   X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
	                     multi_output=True)
	    if y.ndim == 2 and y.shape[1] == 1:
	        y = column_or_1d(y, warn=True)

	    if not incremental:
	        self._label_binarizer = LabelBinarizer32()
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

// Score for BaseMultiLayerPerceptron32 is R2Score or Accuracy depending on LossFuncName
func (mlp *BaseMultilayerPerceptron32) Score(Xmatrix, Ymatrix mat.Matrix) float64 {
	X, Y := ToDense32(Xmatrix), ToDense32(Ymatrix)
	nSamples, nOutputs := X.RawMatrix().Rows, mlp.GetNOutputs()
	Ypred := blas32.General{Rows: nSamples, Cols: nOutputs, Stride: nOutputs, Data: make([]float32, nSamples*nOutputs)}
	mlp.Predict(X, General32(Ypred))
	if mlp.LossFuncName == "square_loss" {
		return float64(r2Score32(blas32.General(Y), Ypred))
	}
	return float64(accuracyScore32(blas32.General(Y), Ypred))
}

// SGDOptimizer32 is the stochastic gradient descent optimizer
type SGDOptimizer32 struct {
	Params           []float32
	LearningRateInit float32
	LearningRate     float32
	PowerT           float32
	LRSchedule       string
	Momentum         float32
	Nesterov         bool
	velocities       []float32
}

func (opt *SGDOptimizer32) iterationEnds(timeStep float32) {
	if strings.EqualFold(opt.LRSchedule, "invscaling") {
		opt.LearningRate = opt.LearningRateInit / M32.Pow(timeStep+1, opt.PowerT)
	}

}
func (opt *SGDOptimizer32) triggerStopping(msg string, verbose bool) bool {
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
func (opt *SGDOptimizer32) updateParams(grads []float32) {
	if opt.velocities == nil {
		opt.velocities = make([]float32, len(grads))
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

// AdamOptimizer32 is the stochastic adam optimizer
type AdamOptimizer32 struct {
	Params                []float32
	LearningRateInit      float32
	LearningRate          float32
	Beta1, Beta2, Epsilon float32
	t                     float32
	ms, vs                []float32
	beta1t, beta2t        float32
}

func (opt *AdamOptimizer32) iterationEnds(timeStep float32)                {}
func (opt *AdamOptimizer32) triggerStopping(msg string, verbose bool) bool { return true }
func (opt *AdamOptimizer32) updateParams(grads []float32) {
	if opt.t == 0 {
		opt.ms = make([]float32, len(grads))
		opt.vs = make([]float32, len(grads))
		opt.beta1t, opt.beta2t = 1, 1
	}
	opt.t++
	for i, grad := range grads {
		opt.ms[i] = opt.Beta1*opt.ms[i] + (1-opt.Beta1)*grad
		opt.vs[i] = opt.Beta2*opt.vs[i] + (1-opt.Beta2)*grad*grad
		opt.beta1t *= opt.Beta1
		opt.beta2t *= opt.Beta2
		opt.LearningRate = opt.LearningRateInit * M32.Sqrt(1-opt.beta2t) / (1. - opt.beta1t)
		update := -opt.LearningRate * opt.ms[i] / (M32.Sqrt(opt.vs[i]) + opt.Epsilon)
		opt.Params[i] += update
	}
}

func toLogits32(ym blas32General) {
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
			M := MaxIdx32(ym.Data[ypos : ypos+ym.Cols])
			for c := 0; c < ym.Cols; c++ {
				v := float32(0)
				if c == M {
					v = 1.
				}
				ym.Data[ypos+c] = v
			}
		}
	}
}

func r2Score32(yTrue, yPred blas32General) float32 {
	var r2acc float32
	for c := 0; c < yTrue.Cols; c++ {
		// yTrueAvg
		var yTrueAvg, yNum, yDen float32
		for r, ypos := 0, 0; r < yTrue.Rows; r, ypos = r+1, ypos+yTrue.Stride {
			yTrueAvg += yTrue.Data[ypos+c]

		}
		yTrueAvg /= float32(yTrue.Rows)
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
	return r2acc / float32(yTrue.Cols)
}

func accuracyScore32(Y, H blas32General) float32 {
	N := float32(0)
	for i, hpos, ypos := 0, 0, 0; i < Y.Rows; i, hpos, ypos = i+1, hpos+H.Stride, ypos+Y.Stride {
		rowEq := true
		for c := 0; c < Y.Cols; c++ {
			rowEq = rowEq && H.Data[hpos+c] == Y.Data[ypos+c]
		}
		if rowEq {
			N++
		}
	}
	return float32(N) / float32(Y.Rows)

}

// SetParams allow settings params from a map. (used by Unmarshal)
func (mlp *BaseMultilayerPerceptron32) SetParams(params map[string]interface{}) {
	r := reflect.Indirect(reflect.ValueOf(mlp))
	for k, v := range params {
		field := r.FieldByNameFunc(func(s string) bool {
			return strings.EqualFold(s, k)
		})
		if field.Kind() != 0 {
			field.Set(reflect.ValueOf(v))
		}
	}
}

// Unmarshal init params intercepts_ coefs_ from json
func (mlp *BaseMultilayerPerceptron32) Unmarshal(buf []byte) error {
	type Map = map[string]interface{}
	mp := Map{}
	err := json.Unmarshal(buf, &mp)
	if err != nil {
		panic(err)
	}
	if params, ok := mp["params"]; ok {
		if pmap, ok := params.(Map); ok {
			mlp.SetParams(pmap)
		}
	} else {
		mlp.SetParams(mp)
	}
	if coefs, ok := mp["coefs_"]; ok {
		intercepts, ok := mp["intercepts_"]
		if !ok {
			return fmt.Errorf("intercepts_ not set")
		}
		intercepts2, ok := intercepts.([]interface{})
		if !ok {
			return fmt.Errorf("intercepts_ must be an array")
		}
		if c64, ok := coefs.([]interface{}); ok {

			if len(c64) == 0 {
				return fmt.Errorf("coefs_ must be non-empty")
			}
			b64coefs := make([]blas64General, len(c64))
			for i := range b64coefs {
				b64coefs[i] = blas64FromInterface(c64[i])
			}
			mlp.NLayers = len(b64coefs) + 1
			mlp.HiddenLayerSizes = make([]int, mlp.NLayers-2)

			NInputs := b64coefs[0].Rows
			mlp.NOutputs = b64coefs[len(b64coefs)-1].Cols
			layerUnits := make([]int, mlp.NLayers)
			layerUnits[0] = NInputs
			packedSize := 0
			for il := range c64 {
				layerUnits[il+1] = b64coefs[il].Cols
				packedSize += (1 + layerUnits[il]) * layerUnits[il+1]
			}
			layerUnits[mlp.NLayers-1] = mlp.NOutputs
			mlp.initialize(mlp.NOutputs, layerUnits, true, mlp.NOutputs > 1)

			for i := 0; i < mlp.NLayers-1; i++ {
				intercept64 := floats64FromInterface(intercepts2[i])
				for off := 0; off < len(mlp.Intercepts[i]); off++ {
					mlp.Intercepts[i][off] = float32(intercept64[off])
				}
				g := General32(mlp.Coefs[i])
				(&g).Copy(General64(b64coefs[i]))
			}
		} else {
			return fmt.Errorf("coefs_ must be [][][]float64, found %T", coefs)
		}
	}
	return err
}

// ToDense32 returns w view of m if m is a RawMatrixer, et returns a dense copy of m
func ToDense32(m Matrix) General32 {
	if d, ok := m.(General32); ok {
		return d
	}
	if m == mat.Matrix(nil) {
		return General32{}
	}
	if rm, ok := m.(RawMatrixer32); ok {
		return General32(rm.RawMatrix())
	}
	ret := General32{}
	ret.Copy(m)
	return ret
}

// FromDense32 fills dst (mat.Mutable) with src (mat.Dense)
func FromDense32(dst Mutable, dense General32) General32 {
	if dst == Mutable(nil) {
		return dense
	}
	src := dense.RawMatrix()
	if rawmatrixer, ok := dst.(RawMatrixer32); ok {
		dstmat := rawmatrixer.RawMatrix()
		if &dstmat.Data[0] == &src.Data[0] {
			return dense
		}
		for r, srcpos, dstpos := 0, 0, 0; r < src.Rows; r, srcpos, dstpos = r+1, srcpos+src.Stride, dstpos+dstmat.Stride {
			for c := 0; c < src.Cols; c++ {
				dstmat.Data[dstpos+c] = src.Data[srcpos+c]
			}
		}
		return dense
	}
	for r, pos := 0, 0; r < src.Rows; r, pos = r+1, pos+src.Stride {
		for c := 0; c < src.Cols; c++ {
			dst.Set(r, c, float64(src.Data[pos+c]))
		}
	}
	return dense
}

// LabelBinarizer32 Binarize labels in a one-vs-all fashion
type LabelBinarizer32 struct {
	NegLabel, PosLabel float32
	Classes            [][]float32
}

// NewLabelBinarizer32 ...
func NewLabelBinarizer32(NegLabel, PosLabel float32) *LabelBinarizer32 {
	return &LabelBinarizer32{NegLabel: NegLabel, PosLabel: PosLabel}
}

// TransformerClone ...
func (m *LabelBinarizer32) TransformerClone() *LabelBinarizer32 {
	clone := *m
	return &clone
}

// Fit for binarizer register classes
func (m *LabelBinarizer32) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	Y := ToDense32(Ymatrix)
	if m.PosLabel == m.NegLabel {
		m.PosLabel += 1.
	}
	y := Y.RawMatrix()
	m.Classes = make([][]float32, y.Cols)
	for j := 0; j < y.Cols; j++ {
		cmap := make(map[float32]bool)
		for i, yi := 0, 0; i < y.Rows; i, yi = i+1, yi+y.Stride {
			yval := y.Data[yi+j]
			if _, present := cmap[yval]; present {
				continue
			}
			cmap[yval] = true
			m.Classes[j] = append(m.Classes[j], yval)
		}
		sort.Sort(Float32Slice(m.Classes[j]))
	}
	return m
}

// Float32Slice implements sort.Interface.
type Float32Slice []float32

func (p Float32Slice) Len() int           { return len(p) }
func (p Float32Slice) Less(i, j int) bool { return p[i] < p[j] || M32.IsNaN(p[i]) && !M32.IsNaN(p[j]) }
func (p Float32Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Transform for LabelBinarizer32
func (m *LabelBinarizer32) Transform(X, Y mat.Matrix) (Xout, Yout General32) {
	Xout = ToDense32(X)
	NSamples, _ := Y.Dims()
	NOutputs := 0
	for _, classes := range m.Classes {
		NOutputs += len(classes)
	}

	Yout = General32{Rows: NSamples, Cols: NOutputs, Stride: NOutputs, Data: make([]float32, NSamples*NOutputs)}
	y, yo := ToDense32(Y).RawMatrix(), Yout.RawMatrix()
	baseCol := 0
	for j := 0; j < y.Cols; j++ {
		cmap := make(map[float32]int)
		for classNo, val := range m.Classes[j] {
			cmap[val] = classNo
		}
		for i, yi, yo0 := 0, 0, 0; i < y.Rows; i, yi, yo0 = i+1, yi+y.Stride, yo0+yo.Stride {
			val := y.Data[yi+j]
			if classNo, ok := cmap[val]; ok {
				yo.Data[yo0+baseCol+classNo] = m.PosLabel
			} else {
				yo.Data[yo0+baseCol+classNo] = m.NegLabel
			}
		}
		baseCol += len(m.Classes[j])
	}
	return
}

// FitTransform fit to data, then transform it
func (m *LabelBinarizer32) FitTransform(X, Y mat.Matrix) (Xout, Yout General32) {
	m.Fit(X, Y)
	return m.Transform(X, Y)
}

// InverseTransform for LabelBinarizer32
func (m *LabelBinarizer32) InverseTransform(X, Y General32) (Xout, Yout General32) {
	Xout = X
	NSamples, _ := Y.Dims()
	NOutputs := len(m.Classes)

	Yout = General32{Rows: NSamples, Cols: NOutputs, Stride: NOutputs, Data: make([]float32, NSamples*NOutputs)}
	y, yo := Y.RawMatrix(), Yout.RawMatrix()
	for j, baseCol := 0, 0; baseCol < y.Cols; j, baseCol = j+1, baseCol+len(m.Classes[j]) {
		for i, yi, yo0 := 0, 0, 0; i < y.Rows; i, yi, yo0 = i+1, yi+y.Stride, yo0+yo.Stride {
			classNo := MaxIdx32(y.Data[yi+baseCol : yi+baseCol+len(m.Classes[j])])

			yo.Data[yo0+j] = m.Classes[j][classNo]
		}
		baseCol += len(m.Classes[j])
	}
	return
}

func isBinarized32(yb blas32.General) bool {
	for r, pos := 0, 0; r < yb.Rows; r, pos = r+1, pos+yb.Stride {
		for c := 0; c < yb.Cols; c++ {
			v := yb.Data[pos+c]
			if v != 0 && v != 1 {
				return false
			}
		}
	}
	return true
}
