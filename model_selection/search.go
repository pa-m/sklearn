package modelselection

import (
	"fmt"
	"reflect"
	"runtime"

	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
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
type GridSearchCV struct {
	Estimator base.Transformer
	ParamGrid map[string][]interface{}
	Scorer    func(Ytrue, Ypred *mat.Dense) float64
	CV        Splitter
	Verbose   bool
	NJobs     int

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
	estCloner := gscv.Estimator.(base.TransformerCloner)
	arrOfMap := ParameterGrid(gscv.ParamGrid)
	gscv.CVResults = make(map[string][]interface{})
	for k := range gscv.ParamGrid {
		gscv.CVResults[k] = make([]interface{}, len(arrOfMap))
	}
	gscv.CVResults["score"] = make([]interface{}, len(arrOfMap))

	type structIn struct {
		cvindex   int
		params    map[string]interface{}
		estimator base.Transformer
		score     float64
	}
	BESTIDX := floats.MaxIdx //FIXME, depends of scorer
	dowork := func(sin structIn) structIn {
		sin.estimator = estCloner.Clone()
		for k, v := range sin.params {
			setParam(sin.estimator, k, v)
		}
		cvres := CrossValidate(sin.estimator, X, Y, nil, gscv.Scorer, gscv.CV, gscv.NJobs)
		sin.score = floats.Sum(cvres.TestScore) / float64(len(cvres.TestScore))
		bestFold := BESTIDX(cvres.TestScore)
		sin.estimator = cvres.Estimator[bestFold]
		return sin
	}
	chin := make(chan structIn)
	chout := make(chan structIn)
	worker := func(j int) {
		for sin := range chin {
			chout <- dowork(sin)
		}
	}
	if gscv.NJobs <= 0 {
		gscv.NJobs = runtime.NumCPU()
	}
	for j := 0; j < gscv.NJobs; j++ {
		go worker(j)
	}
	for cvindex, params := range arrOfMap {
		chin <- structIn{cvindex: cvindex, params: params}
	}
	close(chin)
	gscv.BestIndex = -1
	for range arrOfMap {
		sout := <-chout
		for k, v := range sout.params {
			gscv.CVResults[k][sout.cvindex] = v
		}
		gscv.CVResults["score"][sout.cvindex] = sout.score
		if gscv.BestIndex == -1 || sout.score > gscv.CVResults["score"][gscv.BestIndex].(float64) {
			gscv.BestIndex = sout.cvindex
			gscv.BestEstimator = sout.estimator
			gscv.BestParams = sout.params
			gscv.BestScore = sout.score
		}
	}
	close(chout)

	return gscv
}

// Predict ...
func (gscv *GridSearchCV) Predict(X, Y *mat.Dense) base.Transformer {
	return gscv
}

// Transform ...
func (gscv *GridSearchCV) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	Xout = X
	return
}

func setParam(estimator base.Transformer, k string, v interface{}) {
	est := reflect.ValueOf(estimator)
	est = reflect.Indirect(est)
	if est.Kind().String() != "struct" {
		panic(est.Kind().String())
	}
	if field := est.FieldByName(k); field.Kind() != 0 {
		switch field.Kind() {
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
		default:
			panic(fmt.Errorf("failed to set %s %s to %v", k, field.Type().String(), v))

		}

	} else {
		panic(fmt.Errorf("no field %s in %T", k, estimator))
	}

}
