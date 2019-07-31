package gaussianprocess

import "gonum.org/v1/gonum/mat"

func eye(n int) *mat.DiagDense {
	m := mat.NewDiagDense(n, nil)
	mdata := m.RawSymBand().Data
	for i := 0; i < n; i++ {
		mdata[i] = 1
	}
	return m
}
