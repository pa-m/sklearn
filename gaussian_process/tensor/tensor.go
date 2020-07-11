// Package tensor ...
// very simplified tensor with interface compatible with gorgonia.org/tensor
package tensor

import (
	"errors"
	"reflect"
)

// Dtype is tensor element type
type Dtype = reflect.Kind

// Float64 is element type
var Float64 = reflect.Float64

// Shape ...
type Shape []int

// Dense is a dense tensor
type Dense struct {
	dtype   Dtype
	shape   []int
	strides []int
	data    interface{}
}

// Tensor is the minimal interface for tensor
type Tensor interface {
	Shape() Shape
	At(args ...int) (interface{}, error)
}

// Iterator is the interface for tensor iterator
type Iterator interface {
	Start() (int, error)
	Next() (int, error)
}

// Option can be passed to NewDense ex: WithBacking
type Option func(t *Dense)

// ErrEndReached is returned by iterator when at ane
var ErrEndReached = errors.New("at end")

// ErrIndexOutOfBounds is returned by At when out of bounds
var ErrIndexOutOfBounds = errors.New("index out of bounds")

// DenseIterator is returned by Dense.Iterator
type DenseIterator struct {
	t     *Dense
	index int
}

// Start returns 0
func (it *DenseIterator) Start() (i int, e error) {
	i = it.index
	if i >= len(it.t.data.([]float64)) {
		e = ErrEndReached
	}
	it.index++
	return i, e
}

// Next return current index and advance
func (it *DenseIterator) Next() (i int, e error) {
	i = it.index
	if i >= len(it.t.data.([]float64)) {
		e = ErrEndReached
	}
	it.index++
	return i, e
}

// Dtype returns element type
func (t *Dense) Dtype() Dtype { return t.dtype }

// Shape returns dimensions extend
func (t *Dense) Shape() Shape { return t.shape }

// At returns one element
func (t *Dense) At(args ...int) (v interface{}, e error) {
	var pos int
	for dim, index := range args {
		if index < 0 || index >= t.shape[dim] {
			return nil, ErrIndexOutOfBounds
		}
		pos += index * t.strides[dim]
	}
	data := t.data.([]float64)

	return data[pos], nil
}

// Strides returns position-shift per dimension
func (t *Dense) Strides() []int { return t.strides }

// Data returns underlying data
func (t *Dense) Data() interface{} { return t.data }

// Iterator returns a flat iterator
func (t *Dense) Iterator() Iterator { return &DenseIterator{t: t} }

// NewDense ...
func NewDense(ty Dtype, s Shape, args ...Option) *Dense {
	st := make(Shape, len(s))
	sz := 1
	for i := range s {
		j := len(s) - 1 - i
		st[j] = sz
		sz *= s[j]
	}
	t := &Dense{dtype: ty, shape: s, strides: st}
	for _, o := range args {
		o(t)
	}
	if t.data == nil {
		t.data = make([]float64, sz)
	}
	return t
}

// WithBacking binds an existing storage to tensor
func WithBacking(data []float64) Option { return func(t *Dense) { t.data = data } }
