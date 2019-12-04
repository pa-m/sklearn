package modelselection

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/pa-m/sklearn/base"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// ParameterGrid ...
func ParameterGrid(paramGrid map[string][]interface{}) (out []map[string]interface{}) {
	makeArr := func(name string, values []interface{}, prevArr []map[string]interface{}) (out []map[string]interface{}) {

		if len(prevArr) == 0 {
			for _, v := range values {
				out = append(out, map[string]interface{}{name: v})
			}
		} else {
			for _, map1 := range prevArr {
				for _, v := range values {
					map2 := make(map[string]interface{})
					for k1, v1 := range map1 {
						map2[k1] = v1
					}
					map2[name] = v
					out = append(out, map2)
				}
			}
		}

		return
	}
	for k, v := range paramGrid {
		out = makeArr(k, v, out)
	}
	return
}

// GridSearchCV ...
// Estimator is the base estimator. it must implement base.Predicter
// Scorer is a function  __returning a higher score when Ypred is better__
// CV is a splitter (defaults to KFold)
type GridSearchCV struct {
	Estimator          base.Predicter
	ParamGrid          map[string][]interface{}
	Scorer             func(Ytrue, Ypred mat.Matrix) float64
	CV                 Splitter
	Verbose            bool
	NJobs              int
	LowerScoreIsBetter bool
	UseChannels        bool
	RandomState        rand.Source

	CVResults     map[string][]interface{}
	BestEstimator base.Predicter
	BestScore     float64
	BestParams    map[string]interface{}
	BestIndex     int
	NOutputs      int
}

// PredicterClone ...
func (gscv *GridSearchCV) PredicterClone() base.Predicter {
	if gscv == nil {
		return nil
	}
	clone := *gscv
	if sourceCloner, ok := clone.RandomState.(base.SourceCloner); ok && sourceCloner != base.SourceCloner(nil) {
		clone.RandomState = sourceCloner.SourceClone()
	}
	return &clone
}

// IsClassifier returns underlaying estimater IsClassifier
func (gscv *GridSearchCV) IsClassifier() bool {
	if maybeClf, ok := gscv.Estimator.(base.Predicter); ok {
		return maybeClf.IsClassifier()
	}
	return false
}

// Fit ...
func (gscv *GridSearchCV) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	gscv.NOutputs = Y.RawMatrix().Cols
	isBetter := func(score, refscore float64) bool {
		if gscv.LowerScoreIsBetter {
			return score < refscore
		}
		return score > refscore
	}
	bestIdx := func(scores []float64) int {
		best := -1
		for i, score := range scores {
			if best < 0 || isBetter(score, scores[best]) {
				best = i
			}
		}
		return best
	}

	estCloner := gscv.Estimator
	// get seed for all estimator clone

	paramArray := ParameterGrid(gscv.ParamGrid)

	if gscv.RandomState == rand.Source(nil) {
		gscv.RandomState = base.NewSource(0)
	}
	if gscv.CV == Splitter(nil) {
		gscv.CV = &KFold{NSplits: 3, Shuffle: true, RandomState: gscv.RandomState}
	}
	gscv.CVResults = make(map[string][]interface{})
	for k := range gscv.ParamGrid {
		gscv.CVResults[k] = make([]interface{}, len(paramArray))
	}
	gscv.CVResults["score"] = make([]interface{}, len(paramArray))

	type structIn struct {
		index     int
		params    map[string]interface{}
		estimator base.Predicter
		cv        Splitter
		score     float64
	}
	dowork := func(sin *structIn) {
		cvres := CrossValidate(sin.estimator, X, Y, nil, gscv.Scorer, sin.cv, gscv.NJobs)
		sin.score = floats.Sum(cvres.TestScore) / float64(len(cvres.TestScore))
		bestFold := bestIdx(cvres.TestScore)
		sin.estimator = cvres.Estimator[bestFold]
	}
	gscv.BestIndex = -1

	{
		sin := make([]structIn, len(paramArray))
		for i, params := range paramArray {
			sin[i] = structIn{index: i, params: params, estimator: estCloner.PredicterClone(), cv: gscv.CV.SplitterClone()}
			for k, v := range sin[i].params {
				setParam(sin[i].estimator, k, v)
			}
		}
		base.Parallelize(gscv.NJobs, len(paramArray), func(th, start, end int) {
			for i := start; i < end; i++ {
				dowork(&sin[i])
				for k, v := range paramArray[i] {
					gscv.CVResults[k][i] = v
				}
				gscv.CVResults["score"][i] = sin[i].score
			}
		})
		for i, sout := range sin {
			if gscv.BestIndex == -1 || isBetter(sout.score, gscv.CVResults["score"][gscv.BestIndex].(float64)) {
				gscv.BestIndex = i
				gscv.BestEstimator = sout.estimator
				gscv.BestParams = sout.params
				gscv.BestScore = sout.score
			}
		}
	}

	return gscv
}

// Score for gridSearchCV returns best estimator score
func (gscv *GridSearchCV) Score(X, Y mat.Matrix) float64 {
	return gscv.BestEstimator.Score(X, Y)
}

// GetNOutputs returns output columns number for Y to pass to predict
func (gscv *GridSearchCV) GetNOutputs() int {
	return gscv.NOutputs
}

// Predict ...
func (gscv *GridSearchCV) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	return gscv.BestEstimator.(base.Predicter).Predict(X, Y)
}

func getParam(estimator interface{}, k string) (v interface{}, ok bool) {
	est := reflect.ValueOf(estimator)
	est = reflect.Indirect(est)
	if est.Kind().String() != "struct" {
		panic(est.Kind().String())
	}
	field := est.FieldByNameFunc(func(name string) bool { return strings.EqualFold(name, k) })
	if ok = field.Kind() != 0; ok {
		v = field.Interface()

	}
	return
}

func setParam(estimator base.Predicter, k string, v interface{}) {
	est := reflect.ValueOf(estimator)
	est = reflect.Indirect(est)
	if est.Kind().String() != "struct" {
		panic(est.Kind().String())
	}
	field := est.FieldByNameFunc(func(name string) bool { return strings.EqualFold(name, k) })
	switch field.Kind() {
	case 0:
		panic(fmt.Errorf("no field %s in %T", k, estimator))
	case reflect.String:
		field.SetString(v.(string))
	case reflect.Float64:
		switch vv := v.(type) {
		case int:
			field.SetFloat(float64(vv))
		case float32:
			field.SetFloat(float64(vv))
		case float64:
			field.SetFloat(float64(vv))
		default:
			panic(fmt.Errorf("failed to set %s %s to %v", k, field.Type().String(), v))
		}
	case reflect.Int:
		field.Set(reflect.ValueOf(v))

	case reflect.Interface:
		field.Set(reflect.ValueOf(v))
	default:
		field.Set(reflect.ValueOf(v))
		//panic(fmt.Errorf("failed to set %s %s to %v", k, field.Type().String(), v))

	}

}
