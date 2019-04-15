package modelselection

import (
	// "fmt"
	"runtime"
	"time"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/mat"
)

// CrossValidateResult is the struct result of CrossValidate. it includes TestScore,FitTime,ScoreTime,Estimator
type CrossValidateResult struct {
	TestScore          []float64
	FitTime, ScoreTime []time.Duration
	Estimator          []base.Predicter
}

// Len for CrossValidateResult to implement sort.Interface
func (r CrossValidateResult) Len() int { return len(r.TestScore) }

// Less  for CrossValidateResult to implement sort.Interface
func (r CrossValidateResult) Less(i, j int) bool { return r.TestScore[j] < r.TestScore[i] }

// Swap  for CrossValidateResult to implement sort.Interface
func (r CrossValidateResult) Swap(i, j int) {
	r.TestScore[i], r.TestScore[j] = r.TestScore[j], r.TestScore[i]
	r.FitTime[i], r.FitTime[j] = r.FitTime[j], r.FitTime[i]
	r.ScoreTime[i], r.ScoreTime[j] = r.ScoreTime[j], r.ScoreTime[i]
	r.Estimator[i], r.Estimator[j] = r.Estimator[j], r.Estimator[i]
}

// CrossValidate Evaluate a score by cross-validation
// scorer is a func(Ytrue,Ypred) float64
// only mean_squared_error for now
// NJobs is the number of goroutines. if <=0, runtime.NumCPU is used
func CrossValidate(estimator base.Predicter, X, Y *mat.Dense, groups []int, scorer func(Ytrue, Ypred mat.Matrix) float64, cv Splitter, NJobs int) (res CrossValidateResult) {

	if NJobs <= 0 {
		NJobs = runtime.NumCPU()
	}
	NSplits := cv.GetNSplits(X, Y)
	if NJobs > NSplits {
		NJobs = NSplits
	}
	if cv == Splitter(nil) {
		cv = &KFold{NSplits: 3, Shuffle: true}
	}
	res.Estimator = make([]base.Predicter, NSplits)
	res.TestScore = make([]float64, NSplits)
	res.FitTime = make([]time.Duration, NSplits)
	res.ScoreTime = make([]time.Duration, NSplits)
	type structIn struct {
		iSplit int
		Split
	}
	type structOut struct {
		iSplit int
		score  float64
	}
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()
	processSplit := func(job int, Xjob, Yjob *mat.Dense, sin structIn) structOut {
		Xtrain, Xtest, Ytrain, Ytest := &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}
		// fmt.Println("iSplit", sin.iSplit, "TrainIndex", sin.Split.TrainIndex[:10])
		trainLen, testLen := len(sin.Split.TrainIndex), len(sin.Split.TestIndex)
		Xtrain.SetRawMatrix(base.MatGeneralRowSlice(Xjob.RawMatrix(), 0, trainLen))
		Ytrain.SetRawMatrix(base.MatGeneralRowSlice(Yjob.RawMatrix(), 0, trainLen))
		Xtest.SetRawMatrix(base.MatGeneralRowSlice(Xjob.RawMatrix(), trainLen, trainLen+testLen))
		Ytest.SetRawMatrix(base.MatGeneralRowSlice(Yjob.RawMatrix(), trainLen, trainLen+testLen))
		for i0, i1 := range sin.Split.TrainIndex {
			Xtrain.SetRow(i0, X.RawRowView(i1))
			Ytrain.SetRow(i0, Y.RawRowView(i1))
		}
		for i0, i1 := range sin.Split.TestIndex {
			Xtest.SetRow(i0, X.RawRowView(i1))
			Ytest.SetRow(i0, Y.RawRowView(i1))
		}

		res.Estimator[sin.iSplit] = estimator.PredicterClone()
		t0 := time.Now()
		res.Estimator[sin.iSplit].Fit(Xtrain, Ytrain)
		res.FitTime[sin.iSplit] = time.Since(t0)
		t0 = time.Now()
		Ypred := mat.NewDense(Xtest.RawMatrix().Rows, res.Estimator[sin.iSplit].GetNOutputs(), nil)
		res.Estimator[sin.iSplit].Predict(Xtest, Ypred)
		score := scorer(Ytest, Ypred)
		res.ScoreTime[sin.iSplit] = time.Since(t0)
		//fmt.Printf("score for split %d is %g\n", sin.iSplit, score)
		return structOut{sin.iSplit, score}

	}
	if NJobs > 1 {
		var sin = make([]structIn, 0, NSplits)
		for split := range cv.Split(X, Y) {
			sin = append(sin, structIn{iSplit: len(sin), Split: split})
		}
		base.Parallelize(NJobs, NSplits, func(th, start, end int) {
			var Xjob, Yjob = mat.NewDense(NSamples, NFeatures, nil), mat.NewDense(NSamples, NOutputs, nil)
			for i := start; i < end; i++ {
				sout := processSplit(th, Xjob, Yjob, sin[i])
				res.TestScore[sout.iSplit] = sout.score
			}
		})
	} else { // NJobs==1
		var Xjob, Yjob = mat.NewDense(NSamples, NFeatures, nil), mat.NewDense(NSamples, NOutputs, nil)
		var isplit int
		for split := range cv.Split(X, Y) {
			sout := processSplit(0, Xjob, Yjob, structIn{iSplit: isplit, Split: split})
			res.TestScore[sout.iSplit] = sout.score
			isplit++
		}
	}
	return
}
