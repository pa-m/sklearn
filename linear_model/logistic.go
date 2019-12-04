package linearmodel

import (
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/floats"

	"github.com/pa-m/sklearn/metrics"

	"github.com/pa-m/sklearn/preprocessing"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"

	"gonum.org/v1/gonum/blas/blas64"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/optimize"
)

// LogisticRegression Logistic Regression (aka logit, MaxEnt) classifier.
// In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’,
// and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’.
// This class implements regularized logistic regression using the ‘lbfgs’ solvers.
// support only L2 regularization with primal formulation.
type LogisticRegression struct {
	Alpha         float64          `json:"alpha"`
	MaxIter       int              `json:"max_iter"`
	LossFuncName  string           `json:"loss_func_name"`
	RandomState   base.RandomState `json:"random_state"`
	Tol           float64          `json:"tol"`
	Verbose       bool             `json:"verbose"`
	NIterNoChange int              `json:"n_iter_no_change"`

	// Outputs
	NLayers       int
	NIter         int
	NOutputs      int
	Intercept     []float64      `json:"intercepts_"`
	Coef          blas64.General `json:"coefs_"`
	OutActivation string         `json:"out_activation_"`
	Loss          float64

	// internal
	t                  int
	LossCurve          []float64
	BestLoss           float64
	NoImprovementCount int
	InterceptsGrads    []float64
	CoefsGrads         blas64.General
	packedParameters   []float64
	packedGrads        []float64
	// bestParameters     []float64
	lb             *preprocessing.LabelBinarizer
	beforeMinimize func(optimize.Problem, []float64)
}

// logregActivation is a map containing the inplace_activation functions
var logregActivation = map[string]func(z blas64.General){
	"logistic": func(z blas64.General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] = 1 / (1 + math.Exp(-z.Data[zpos+col]))
			}
		}
	},
	"relu": func(z blas64.General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			for col := 0; col < z.Cols; col++ {
				if z.Data[zpos+col] < 0 {
					z.Data[zpos+col] = 0
				}
			}
		}
	},
	"softmax": func(z blas64.General) {
		for row, zpos := 0, 0; row < z.Rows; row, zpos = row+1, zpos+z.Stride {
			sum := float64(0)
			for col := 0; col < z.Cols; col++ {

				z.Data[zpos+col] = math.Exp(z.Data[zpos+col])
				sum += z.Data[zpos+col]
			}
			for col := 0; col < z.Cols; col++ {
				z.Data[zpos+col] /= sum
			}
		}
	},
}

// logregLossFunctions is a map for loss functions
var logregLossFunctions = map[string]func(y, h blas64.General) float64{
	"log_loss": func(y, h blas64.General) float64 {
		sum := float64(0)
		hmin, hmax := math.Nextafter(0, 1), math.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				if y.Data[ypos+col] != 0 {
					sum += -y.Data[ypos+col] * math.Log(hval)
				}
			}
		}
		return sum / float64(h.Rows)
	},
	"binary_log_loss": func(y, h blas64.General) float64 {
		sum := float64(0)
		hmin, hmax := math.Nextafter(0, 1), math.Nextafter(1, 0)
		for row, hpos, ypos := 0, 0, 0; row < y.Rows; row, hpos, ypos = row+1, hpos+h.Stride, ypos+y.Stride {
			for col := 0; col < y.Cols; col++ {
				hval := h.Data[hpos+col]
				if hval < hmin {
					hval = hmin
				} else if hval > hmax {
					hval = hmax
				}
				sum += -y.Data[ypos+col]*math.Log(hval) - (1-y.Data[ypos+col])*math.Log1p(-hval)
			}
		}
		return sum / float64(h.Rows)
	},
}

func addIntercepts64(a blas64.General, b []float64) {
	for arow, apos := 0, 0; arow < a.Rows; arow, apos = arow+1, apos+a.Stride {
		for c := 0; c < a.Cols; c++ {
			a.Data[apos+c] += b[c]
		}
	}
}

func matRowMean64(a blas64.General, b []float64) {
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

// NewLogisticRegression returns a LogisticRegression with defaults: Alpha=1/C=1; Tol=1e-4
func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{

		Alpha:   1,
		MaxIter: 200,
		//RandomState        base.Source,
		Tol:           1e-4,
		Verbose:       false,
		NIterNoChange: 10,
	}
}

// PredicterClone ...
func (m *LogisticRegression) PredicterClone() base.Predicter {
	clone := *m
	if sc, ok := m.RandomState.(base.SourceCloner); ok {
		clone.RandomState = sc.SourceClone()
	}
	return &clone
}

// forwardPass Perform a forward pass on the network by computing the values
// of the neurons the output layer.
//        activations : []blas64.General, length = nLayers - 1
func (m *LogisticRegression) forwardPass(activations []blas64.General) {
	hiddenActivation := logregActivation["logistic"]
	var i int
	for i = 0; i < m.NLayers-1; i++ {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, activations[i], m.Coef, 0, activations[i+1])
		addIntercepts64(activations[i+1], m.Intercept)
		// For the hidden layers
		if (i + 1) != (m.NLayers - 1) {
			hiddenActivation(activations[i+1])
		}
	}
	i = m.NLayers - 2
	// # For the last layer
	outputActivation := logregActivation[m.OutActivation]
	outputActivation(activations[i+1])
}

func (m *LogisticRegression) sumCoefSquares() float64 {
	s := float64(0)

	for _, co := range m.Coef.Data {
		s += co * co
	}

	return s
}

// computeLossGrad Compute the gradient of loss with respect to Coef and intercept for specified layer.
// This function does backpropagation for the specified one layer.
func (m *LogisticRegression) computeLossGrad(layer, NSamples int, activations []blas64.General, deltas, coefGrads blas64.General, interceptGrads []float64) {
	// coefGrads[layer] = safeSparseDot(activations[layer].T, deltas[layer])
	// coefGrads[layer] += (self.alpha * self.coefs_[layer])
	// coefGrads[layer] /= nSamples
	blas64.Gemm(blas.Trans, blas.NoTrans, 1/float64(NSamples), activations[layer], deltas, 0, coefGrads)
	blas64.Axpy(m.Alpha/float64(NSamples), blas64.Vector{N: len(m.Coef.Data), Data: m.Coef.Data, Inc: 1}, blas64.Vector{N: len(coefGrads.Data), Data: coefGrads.Data, Inc: 1})
	// interceptGrads[layer] = np.mean(deltas[layer], 0)
	matRowMean64(deltas, interceptGrads)
}

// backprop Compute the m loss function and its corresponding derivatives with respect to each parameter: weights and bias vectors.
// X : blas64.General shape (nSamples, nFeatures)
// Y : blas64.General shape (nSamples, nOutputs)
// activations : []blas64.General, length=2
// deltas : blas64.General
// coefGrads : blas64.General
// interceptGrads : []float64
func (m *LogisticRegression) backprop(X, y blas64.General, activations []blas64.General, deltas, coefGrads blas64.General, interceptGrads []float64) float64 {
	nSamples := X.Rows
	m.forwardPass(activations)

	//# Get loss
	lossFuncName := m.LossFuncName
	if strings.EqualFold(lossFuncName, "log_loss") && strings.EqualFold(m.OutActivation, "logistic") {
		lossFuncName = "binary_log_loss"
	}
	// y may have less rows than activations il last batch
	loss := logregLossFunctions[lossFuncName](y, activations[len(activations)-1])
	// # Add L2 regularization term to loss
	loss += (0.5 * m.Alpha) * m.sumCoefSquares() / float64(nSamples)

	//# Backward propagate
	last := m.NLayers - 2

	// # The calculation of delta[last] here works with following
	// # combinations of output activation and loss function:
	// # sigmoid and binary cross entropy, softmax and categorical cross
	// # entropy, and identity with squared loss
	//deltas[last] = activations[len(activations)-1] - y

	// y may have less rows than activations il last batch
	{
		H := activations[len(activations)-1]
		D := deltas
		for r, pos := 0, 0; r < y.Rows; r, pos = r+1, pos+y.Stride {
			for o, posc := 0, pos; o < y.Cols; o, posc = o+1, posc+1 {
				D.Data[posc] = H.Data[posc] - y.Data[posc]
			}
		}
	}

	//# Compute gradient for the last layer
	m.computeLossGrad(
		last, nSamples, activations, deltas, coefGrads, interceptGrads)

	return loss
}

// initialize compute size of coef and grad, allocates them and fill Coef with Glorot et Al initialization
func (m *LogisticRegression) initialize(yCols int, layerUnits []int, isMultiClass bool) {
	// # set all attributes, allocate weights etc for first call
	// # Initialize parameters
	m.NIter = 0
	m.t = 0
	m.NOutputs = yCols

	//# Compute the number of layers
	m.NLayers = len(layerUnits)

	//# Output for regression
	if isMultiClass {
		m.OutActivation = "softmax"
		m.LossFuncName = "log_loss"
		//# Output for binary class and multi-label
	} else {
		m.OutActivation = "logistic"
		m.LossFuncName = "binary_log_loss"
	}
	//# Initialize coefficient and intercept layers
	off := (1 + layerUnits[0]) * layerUnits[1]

	mem := make([]float64, 2*off)
	m.packedParameters = mem[0:off]
	m.packedGrads = mem[off : 2*off]

	off = 0
	if m.RandomState == (base.RandomState)(nil) {
		m.RandomState = base.NewLockedSource(uint64(time.Now().UnixNano()))
	}
	type Float64er interface {
		Float64() float64
	}
	var rndFloat64 func() float64
	if float64er, ok := m.RandomState.(Float64er); ok {
		rndFloat64 = float64er.Float64
	} else {
		rndFloat64 = rand.New(m.RandomState).Float64
	}
	prevOff := off
	m.Intercept = mem[off : off+layerUnits[1]]
	off += layerUnits[1]
	m.Coef = blas64.General{Rows: layerUnits[0], Cols: layerUnits[1], Stride: layerUnits[1], Data: mem[off : off+layerUnits[0]*layerUnits[1]]}
	off += layerUnits[0] * layerUnits[1]
	// # Use the initialization method recommended by
	// # Glorot et al.
	factor := float64(6.)
	fanIn, fanOut := layerUnits[0], layerUnits[1]
	factor = 2.

	initBound := math.Sqrt(factor / float64(fanIn+fanOut))
	for pos := prevOff; pos < off; pos++ {
		mem[pos] = rndFloat64() * initBound
	}

	m.InterceptsGrads = mem[off : off+layerUnits[1]]
	off += layerUnits[1]
	m.CoefsGrads = blas64.General{Rows: layerUnits[0], Cols: layerUnits[1], Stride: layerUnits[1], Data: mem[off : off+layerUnits[0]*layerUnits[1]]}
	off += layerUnits[0] * layerUnits[1]

	m.BestLoss = math.Inf(1)
}

// IsClassifier return true if LossFuncName is not square_loss
func (m *LogisticRegression) IsClassifier() bool {
	return true
}

// Fit compute Coef and Intercept
func (m *LogisticRegression) Fit(X, Y mat.Matrix) base.Fiter {
	var xb, yb *mat.Dense
	if xg, ok := X.(*mat.Dense); ok {
		if yg, ok := Y.(*mat.Dense); ok {
			xb, yb = xg, yg
		}
	} else {
		xb = mat.DenseCopyOf(X)
		yb = mat.DenseCopyOf(Y)
	}
	if m.IsClassifier() && !isBinarized(yb) {
		m.lb = preprocessing.NewLabelBinarizer(0, 1)
		xbin, ybin := m.lb.FitTransform(X, Y)
		xb, yb = xbin, ybin
	}
	// # Validate input parameters.
	m.validateHyperparameters()

	x, y := xb.RawMatrix(), yb.RawMatrix()
	nSamples, nFeatures := x.Rows, x.Cols

	m.NOutputs = y.Cols
	layerUnits := []int{nFeatures}
	layerUnits = append(layerUnits, m.NOutputs)

	if m.RandomState == nil {
		m.RandomState = rand.New(base.NewLockedSource(uint64(time.Now().UnixNano())))
	}
	m.initialize(y.Cols, layerUnits, y.Cols > 1)

	// # Initialize lists
	batchSize := nSamples
	activations := []blas64.General{x}
	nFanOut := layerUnits[1]
	activations = append(activations, blas64.General{Rows: batchSize, Cols: nFanOut, Stride: nFanOut, Data: make([]float64, batchSize*nFanOut)})
	deltas := blas64.General{Rows: batchSize, Cols: nFanOut, Stride: nFanOut, Data: make([]float64, batchSize*nFanOut)}

	// # Run the LBFGS solver
	m.fitLbfgs(x, y, activations, deltas, m.CoefsGrads,
		m.InterceptsGrads, layerUnits)
	return m
}

// GetNOutputs returns output columns number for Y to pass to predict
func (m *LogisticRegression) GetNOutputs() int {
	if m.lb != nil {
		return len(m.lb.Classes)
	}
	return m.NOutputs
}

func (m *LogisticRegression) validateHyperparameters() {
	if m.MaxIter <= 0 {
		log.Panicf("maxIter must be > 0, got %d.", m.MaxIter)
	}
	if m.Alpha < 0.0 {
		log.Panicf("alpha must be >= 0, got %g.", m.Alpha)
	}
	if m.NIterNoChange <= 0 {
		log.Panicf("nIterNoChange must be > 0, got %d.", m.NIterNoChange)
	}
}

func (m *LogisticRegression) fitLbfgs(X, y blas64.General, activations []blas64.General, deltas, coefGrads blas64.General,
	interceptGrads []float64, layerUnits []int) {
	method := &optimize.LBFGS{}
	settings := &optimize.Settings{
		FuncEvaluations: m.MaxIter,
		Converger: &optimize.FunctionConverge{
			Relative:   float64(m.Tol),
			Iterations: m.NIterNoChange,
		},
	}

	var mu sync.Mutex // sync access to m.Loss on LossCurve
	problem := optimize.Problem{
		Func: func(w []float64) float64 {
			for i := range w {
				m.packedParameters[i] = float64(w[i])
			}
			loss := float64(m.backprop(X, y, activations, deltas, coefGrads, interceptGrads))
			mu.Lock()
			m.Loss = float64(loss)
			m.LossCurve = append(m.LossCurve, m.Loss)
			m.BestLoss = math.Min(m.BestLoss, m.Loss)
			mu.Unlock()
			return loss
		},
		Grad: func(g, w []float64) {
			// Grad is called just after Func with same w
			if g == nil { // g is nil at first call
				g = make([]float64, len(w))
			}
			for i := range w {
				g[i] = float64(m.packedGrads[i])
			}
		},
	}
	w := make([]float64, len(m.packedParameters))
	for i := range w {
		w[i] = float64(m.packedParameters[i])
	}
	if m.beforeMinimize != nil {
		m.beforeMinimize(problem, w)
	}
	res, err := optimize.Minimize(problem, w, settings, method)
	if err != nil {
		log.Panic(err)
	}
	if res.Status != optimize.GradientThreshold && res.Status != optimize.FunctionConvergence {
		log.Printf("lbfgs optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet.\n", m.MaxIter)
	}
}

// PredictProbas return probability estimates.
// The returned estimates for all classes are ordered by the label of classes.
func (m *LogisticRegression) PredictProbas(Xmatrix mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	X, Y := base.ToDense(Xmatrix).RawMatrix(), base.ToDense(Ymutable)
	if Y.IsEmpty() {
		fanOut := 0
		if m.lb != nil {
			for _, classes := range m.lb.Classes {
				fanOut += len(classes)
			}
		} else {
			fanOut = m.GetNOutputs()
		}
		Y = mat.NewDense(X.Rows, fanOut, nil)
	}

	// # Initialize layer
	activations := []blas64.General{X, Y.RawMatrix()}
	// # forward propagate
	m.forwardPass(activations)
	return base.FromDense(Ymutable, Y)
}

// Predict do forward pass and fills Y (Y must be mat.Mutable)
func (m *LogisticRegression) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	ybin := m.PredictProbas(X, nil)
	var Yclasses *mat.Dense
	if m.lb != nil {
		_, Yclasses = m.lb.InverseTransform(nil, ybin)
	} else if m.IsClassifier() {
		toLogits(ybin.RawMatrix())
		Yclasses = ybin
	}

	return base.FromDense(Y, Yclasses)
}

// Score for LogisticRegression is accuracy
func (m *LogisticRegression) Score(Xmatrix, Ymatrix mat.Matrix) float64 {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	nSamples, nOutputs := X.RawMatrix().Rows, m.GetNOutputs()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	m.Predict(X, Ypred)
	return metrics.AccuracyScore(Y, Ypred, true, nil)
}

func toLogits(ym blas64.General) {
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
			M := floats.MaxIdx(ym.Data[ypos : ypos+ym.Cols])
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

func isBinarized(Y *mat.Dense) bool {
	yb := Y.RawMatrix()
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

/*
func rowSlice(X blas64.General, i, k int) blas64.General {
	var tmp mat.Dense
	(&tmp).SetRawMatrix(X)
	return tmp.Slice(i, k, 0, X.Cols).(*mat.Dense).RawMatrix()
}
*/
