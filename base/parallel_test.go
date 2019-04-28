package base

import (
	"fmt"
	"testing"
)

func TestParalellize(t *testing.T) {
	var a [40]float64
	Parallelize(4, 40, func(th, start, end int) {
		for i := start; i < end; i++ {
			a[i] = float64(th*100 + i)
		}
	})
	expected := "[0 1 2 3 4 5 6 7 8 9 110 111 112 113 114 115 116 117 118 119 220 221 222 223 224 225 226 227 228 229 330 331 332 333 334 335 336 337 338 339]"
	actual := fmt.Sprintf("%g", a)
	if expected != actual {
		t.Errorf("Expected:\n%s\ngot:\n%s", expected, actual)
	}
}
