package modelselection

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/pa-m/sklearn/base"
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
// Estimator is the base estimator. it must implement base.TransformerCloner
// Scorer is a function  __returning a higher score when Ypred is better__
// CV is a splitter (defaults to KFold)
type GridSearchCV struct {
	Estimator          base.Transformer
	ParamGrid          map[string][]interface{}
	Scorer             func(Ytrue, Ypred *mat.Dense) float64
	CV                 Splitter
	Verbose            bool
	NJobs              int
	LowerScoreIsBetter bool
	UseChannels        bool

	CVResults     map[string][]interface{}
	BestEstimator base.Transformer
	BestScore     float64
	BestParams    map[string]interface{}
	BestIndex     int
}

// Clone ...
func (gscv *GridSearchCV) Clone() base.Transformer {
	clone := *gscv
	return &clone
}

// Fit ...
func (gscv *GridSearchCV) Fit(X, Y *mat.Dense) base.Transformer {

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

	estCloner := gscv.Estimator.(base.TransformerCloner)
	// get seed for all estimator clone

	type ClonableRandomState interface {
		Clone() base.Source
	}
	var clonableRandomState ClonableRandomState
	if rs, ok := getParam(gscv.Estimator, "RandomState"); ok {
		if rs1, ok := rs.(ClonableRandomState); ok {
			clonableRandomState = rs1
		}
	}

	paramArray := ParameterGrid(gscv.ParamGrid)
	gscv.CVResults = make(map[string][]interface{})
	for k := range gscv.ParamGrid {
		gscv.CVResults[k] = make([]interface{}, len(paramArray))
	}
	gscv.CVResults["score"] = make([]interface{}, len(paramArray))

	type structIn struct {
		cvindex   int
		params    map[string]interface{}
		estimator base.Transformer
		score     float64
	}
	dowork := func(sin structIn) structIn {

		sin.estimator = estCloner.Clone()

		if clonableRandomState != ClonableRandomState(nil) {

			//setParam(sin.estimator, "RandomState", rand.New(base.NewLockedSource(clonesSeed)))
			setParam(sin.estimator, "RandomState", clonableRandomState)
		}

		for k, v := range sin.params {
			setParam(sin.estimator, k, v)
		}
		CV := gscv.CV.Clone()
		cvres := CrossValidate(sin.estimator, X, Y, nil, gscv.Scorer, CV, gscv.NJobs)
		sin.score = floats.Sum(cvres.TestScore) / float64(len(cvres.TestScore))
		bestFold := bestIdx(cvres.TestScore)
		sin.estimator = cvres.Estimator[bestFold]
		return sin
	}
	gscv.BestIndex = -1

	/*if gscv.UseChannels { // use channels
		chin := make(chan structIn)
		chout := make(chan structIn)
		worker := func(j int) {
			for sin := range chin {
				chout <- dowork(sin)
			}
		}
		if gscv.NJobs <= 0 || gscv.NJobs > runtime.NumCPU() {
			gscv.NJobs = runtime.NumCPU()
		}
		for j := 0; j < gscv.NJobs; j++ {
			go worker(j)
		}
		for cvindex, params := range paramArray {
			chin <- structIn{cvindex: cvindex, params: params}
		}
		close(chin)
		for range paramArray {
			sout := <-chout
			for k, v := range sout.params {
				gscv.CVResults[k][sout.cvindex] = v
			}
			gscv.CVResults["score"][sout.cvindex] = sout.score
			if gscv.BestIndex == -1 || isBetter(sout.score, gscv.CVResults["score"][gscv.BestIndex].(float64)) {
				gscv.BestIndex = sout.cvindex
				gscv.BestEstimator = sout.estimator
				gscv.BestParams = sout.params
				gscv.BestScore = sout.score
			}
		}
		close(chout)

	} else*/{ // use sync.workGroup
		sin := make([]structIn, len(paramArray))
		for i, params := range paramArray {
			sin[i] = structIn{cvindex: i, params: params, estimator: gscv.Estimator}
		}
		base.Parallelize(gscv.NJobs, len(paramArray), func(th, start, end int) {
			for i := start; i < end; i++ {
				sin[i] = dowork(sin[i])
				gscv.CVResults["score"][sin[i].cvindex] = sin[i].score
			}
		})
		for _, sout := range sin {
			if gscv.BestIndex == -1 || isBetter(sout.score, gscv.CVResults["score"][gscv.BestIndex].(float64)) {
				gscv.BestIndex = sout.cvindex
				gscv.BestEstimator = sout.estimator
				gscv.BestParams = sout.params
				gscv.BestScore = sout.score
			}
		}
	}

	return gscv
}

// Predict ...
func (gscv *GridSearchCV) Predict(X, Y *mat.Dense) {
	gscv.BestEstimator.(base.Predicter).Predict(X, Y)
}

// Transform ...
func (gscv *GridSearchCV) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout = X
	return
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

func setParam(estimator base.Transformer, k string, v interface{}) {
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
