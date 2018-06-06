package base

import (
	"reflect"

	"gonum.org/v1/gonum/mat"
)

type float = float64

func unused(...interface{}) {}

// BaseEstimator support may be added in the future
type BaseEstimator interface {
	GetParams() map[string]interface{}
	SetParams(interface{})
}

// Transformer transforms X,Y
type Transformer interface {
	Fit(X, Y *mat.Dense) Transformer
	Transform(X, T *mat.Dense) (Xout, Yout *mat.Dense)
}

// Regressor is the common interface for all regressors
type Regressor interface {
	Transformer
	Predict(X, Y *mat.Dense) Regressor
	Score(X, T *mat.Dense) float64
}

// Classifier is the common interface for all Classifiers
type Classifier struct {
	Regressor
}

// CopyStruct create an new *struct with copied fields using reflection. it's not a deep copy.
func CopyStruct(m interface{}) interface{} {

	mstruct := reflect.ValueOf(m)
	if mstruct.Kind() == reflect.Ptr {
		mstruct = mstruct.Elem()
	}
	m2 := reflect.New(mstruct.Type())
	for i := 0; i < mstruct.NumField(); i++ {
		c := m2.Elem().Type().Field(i).Name[0]
		if m2.Elem().Field(i).CanSet() && c >= 'A' && c <= 'Z' {
			m2.Elem().Field(i).Set(mstruct.Field(i))
		}
	}
	return m2.Interface()
}
