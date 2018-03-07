package base

import (
	"reflect"
)

type float = float64

func unused(...interface{}) {}

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
