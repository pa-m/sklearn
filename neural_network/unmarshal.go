package neuralnetwork

func floats64FromInterface(in interface{}) []float64 {
	t1 := in.([]interface{})
	t2 := make([]float64, len(t1))
	for i := range t1 {
		t2[i] = t1[i].(float64)
	}
	return t2
}

func blas64FromInterface(in interface{}) blas64General {
	t1 := in.([]interface{})
	rows := len(t1)
	t10 := t1[0].([]interface{})
	cols := len(t10)
	t2 := blas64General{Rows: rows, Cols: cols, Stride: cols, Data: make([]float64, rows*cols)}
	for r, pos := 0, 0; r < rows; r, pos = r+1, pos+t2.Stride {
		t3 := floats64FromInterface(t1[r])
		for c := 0; c < cols; c++ {
			t2.Data[pos+c] = t3[c]
		}

	}
	return t2
}
