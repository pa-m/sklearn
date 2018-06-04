package linearModel

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// coordinate descent algorithm for Elastic-Net
// v https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/linear_model/cd_fast.pyx
func enetCoordinateDescent(w *mat.VecDense, alpha, beta float64, X *mat.Dense, Y *mat.VecDense, maxIter int, tol float64, rng *rand.Rand, random, positive bool) (gap, tolyy float64, nIter int) {
	/*
	   coordinate descent algorithm
	       for Elastic-Net regression
	       We minimize
	       (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2
	*/
	gap = tol + 1.
	dwtol := tol

	_, NFeatures := X.Dims()
	R := &mat.VecDense{}
	rmX := X.RawMatrix()
	// # norm_cols_X = (np.asarray(X) ** 2).sum(axis=0)
	normColsX := make([]float64, NFeatures)
	for r, rp := 0, 0; r < rmX.Rows; r, rp = r+1, rp+rmX.Stride {
		for c := 0; c < rmX.Cols; c++ {
			e := rmX.Data[rp+c]
			normColsX[c] += e * e
		}
	}
	// # R = Y - np.dot(X, W.T)
	R.MulVec(X, w)
	R.SubVec(Y, R)

	// # tol = tol * linalg.norm(Y, ord='fro') ** 2
	{
		Y2 := &mat.VecDense{}
		Y2.MulElemVec(Y, Y)
		tol *= mat.Sum(Y2)
	}

	var wii, dwii, wmax, dwmax, dualNormXtA, cons, l1norm float64
	tmp := &mat.VecDense{}
	XtA := &mat.VecDense{}
	RNorm2 := &mat.VecDense{}
	wNorm2 := &mat.VecDense{}
	RY := &mat.VecDense{}

	fsign := func(x float64) float64 {
		if x >= 0 {
			return 1.
		}
		return -1
	}

	for nIter = 0; nIter < maxIter; nIter++ {
		wmax, dwmax = 0., 0.
		var ii int
		for fIter := 0; fIter < NFeatures; fIter++ {
			if random {
				if rng != nil {
					ii = rng.Intn(NFeatures)
				} else {
					ii = rand.Intn(NFeatures)
				}
			} else {
				ii = fIter
			}
			if normColsX[ii] == 0. {
				continue
			}
			// # w_ii = W[:, ii] # Store previous value
			wii = w.AtVec(ii) // store previous value

			if wii != 0. {
				// # R += w_ii * X[:,ii]
				R.AddScaledVec(R, wii, X.ColView(ii))
			}

			// # tmp = (X[:,ii]*R).sum()

			tmp.MulVec(X.ColView(ii).T(), R)

			if positive && tmp.AtVec(0) < 0. {
				w.SetVec(ii, 0)
			} else {
				w.SetVec(ii, (fsign(tmp.AtVec(0))*math.Max(math.Abs(tmp.AtVec(0))-alpha, 0))/(normColsX[ii]+beta))
			}

			if w.AtVec(ii) != 0. {
				// # R -=  w[ii] * X[:,ii] # Update residual
				R.AddScaledVec(R, -w.AtVec(ii), X.ColView(ii))
			}
			// # update the maximum absolute coefficient update
			dwii = math.Abs(w.AtVec(ii) - wii)

			if dwii > dwmax {
				dwmax = dwii
			}
			if math.Abs(w.AtVec(ii)) > wmax {
				wmax = math.Abs(w.AtVec(ii))
			}

		}
		if wmax == 0. || dwmax/wmax < dwtol || nIter == maxIter-1 {
			// # the biggest coordinate update of this iteration was smaller
			// # than the tolerance: check the duality gap as ultimate
			// # stopping criterion

			// # XtA = np.dot(X.T, R) - beta * w
			XtA.MulVec(X.T(), R)
			XtA.AddScaledVec(XtA, -beta, w)
			if positive {
				dualNormXtA = math.Max(float64(NFeatures), mat.Max(XtA))
			} else {
				for i := 0; i < XtA.Len(); i++ {
					XtA.SetVec(i, math.Abs(XtA.AtVec(i)))
				}
				dualNormXtA = math.Max(float64(NFeatures), mat.Max(XtA))
			}
			// # R_norm2 = np.dot(R, R)
			RNorm2.MulElemVec(R, R)
			// # w_norm2 = np.dot(w, w)
			wNorm2.MulElemVec(w, w)

			if dualNormXtA > alpha {
				cons = alpha / dualNormXtA
				ANorm2 := RNorm2.AtVec(0) * cons * cons
				gap = .5 * (RNorm2.AtVec(0) + ANorm2)
			} else {
				cons = 1.
				gap = RNorm2.AtVec(0)
			}
			l1norm = mat.Norm(w, 1)
			RY.MulElemVec(R, Y)
			gap += (alpha*l1norm - cons*mat.Sum(RY)) + .5*beta*(1.+cons*cons)*wNorm2.AtVec(0)
			if gap < tol {
				// # return if we reached desired tolerance
				break
			}
		}

	}
	return gap, tol, nIter + 1
}

func enetCoordinateDescentMultiTask(w *mat.Dense, l1reg, l2reg float64, X *mat.Dense, Y *mat.Dense, maxIter int, tol float64, rng *rand.Rand, random, positive bool) (gap, tolout float64, nIter int) {
	/*
	   coordinate descent algorithm
	       for Elastic-Net regression
	       We minimize
	       (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2
	*/
	gap = tol + 1.
	dwtol := tol

	NSamples, NFeatures := X.Dims()
	_ = NSamples
	_, NTasks := Y.Dims()

	R := &mat.Dense{}
	rmX := X.RawMatrix()
	// # norm_cols_X = (np.asarray(X) ** 2).sum(axis=0)
	normColsX := make([]float64, NFeatures)
	for r, rp := 0, 0; r < rmX.Rows; r, rp = r+1, rp+rmX.Stride {
		for c := 0; c < rmX.Cols; c++ {
			e := rmX.Data[rp+c]
			normColsX[c] += e * e
		}
	}
	// # R = Y - np.dot(X, W.T)
	R.Mul(X, w)
	R.Sub(Y, R)

	// # tol = tol * linalg.norm(Y, ord='fro') ** 2
	{
		Y2 := &mat.Dense{}
		Y2.MulElem(Y, Y)
		tol *= mat.Sum(Y2)
	}

	var dwii, wmax, dwmax, dualNormXtA, cons, XtAaxis1norm, wiiabsmax, l21norm float64
	var ii, fIter int
	tmp := mat.NewVecDense(NTasks, nil)
	wii := mat.NewVecDense(NFeatures, nil)

	XtA := &mat.Dense{}
	RNorm, wNorm, ANorm, nn := 0., 0., 0., 0.
	RY := &mat.Dense{}
	xw := &mat.Dense{}

	for nIter = 0; nIter < maxIter; nIter++ {
		wmax, dwmax = 0., 0.

		for fIter = 0; fIter < NFeatures; fIter++ {
			if random {
				if rng != nil {
					ii = rng.Intn(NFeatures)
				} else {
					ii = rand.Intn(NFeatures)
				}
			} else {
				ii = fIter
			}
			if normColsX[ii] == 0. {
				continue
			}
			// # w_ii = W[:, ii] # Store previous value
			wii.CopyVec(w.RowView(ii)) // store previous value

			//  if np.sum(w_ii ** 2) != 0.0:  # can do better
			if mat.Norm(wii, 2) != 0. {
				// # R +=  X[:,ii] *wii # rank 1 update

				xw.Mul(X.ColView(ii), wii.T())
				R.Add(R, xw)
			}

			// # tmp = np.dot(X[:, ii][None, :], R).ravel()
			tmp.MulVec(R.T(), X.ColView(ii))

			// # nn = sqrt(np.sum(tmp ** 2))
			nn = mat.Norm(tmp, 2)
			// # W[:, ii] = tmp * fmax(1. - l1_reg / nn, 0) / (norm_cols_X[ii] + l2_reg)
			w.RowView(ii).(*mat.VecDense).ScaleVec(math.Max(1.-l1reg/nn, 0)/(normColsX[ii]+l2reg), tmp)

			// # if np.sum(W[:, ii] ** 2) != 0.0:  # can do better

			if mat.Norm(w.RowView(ii), 2) != 0. {
				// # R -= np.dot(X[:, ii][:, None], W[:, ii][None, :])
				// # Update residual : rank 1 update
				xw.Mul(X.ColView(ii), w.RowView(ii).T())
				R.Sub(R, xw)
			}
			// # update the maximum absolute coefficient update
			dwii = 0.
			for o := 0; o < NTasks; o++ {
				v := math.Abs(w.At(ii, o) - wii.AtVec(o))
				if v > dwii {
					dwii = v
				}
			}

			if dwii > dwmax {
				dwmax = dwii
			}
			wiiabsmax = mat.Norm(w, math.Inf(1))
			if wiiabsmax > wmax {
				wmax = wiiabsmax
			}

		}
		if wmax == 0. || dwmax/wmax < dwtol || nIter == maxIter-1 {
			// # the biggest coordinate update of this iteration was smaller
			// # than the tolerance: check the duality gap as ultimate
			// # stopping criterion

			// # XtA = np.dot(X.T, R) - l2_reg * W.T
			XtA.Mul(X.T(), R)
			XtAmat := XtA.RawMatrix()
			Wmat := w.RawMatrix()
			for jXtA, jW := 0, 0; jXtA < XtAmat.Rows*XtAmat.Stride; jXtA, jW = jXtA+XtAmat.Stride, jW+Wmat.Stride {
				for i := range XtAmat.Data[jXtA : jXtA+XtAmat.Cols] {
					XtAmat.Data[jXtA+i] -= l2reg * Wmat.Data[jW+i]
				}
			}
			// # dual_norm_XtA = np.max(np.sqrt(np.sum(XtA ** 2, axis=1)))
			dualNormXtA = 0.
			for ii := 0; ii < NFeatures; ii++ {
				XtAaxis1norm = mat.Norm(XtA.RowView(ii), 2)
				if XtAaxis1norm > dualNormXtA {
					dualNormXtA = XtAaxis1norm
				}
			}
			// # R_norm = linalg.norm(R, ord='fro')
			// # w_norm = linalg.norm(W, ord='fro')
			RNorm = mat.Norm(R, 2)
			wNorm = mat.Norm(w, 2)
			if dualNormXtA > l1reg {
				cons = l1reg / dualNormXtA
				ANorm = RNorm * cons
				gap = .5 * (RNorm*RNorm + ANorm*ANorm)
			} else {
				cons = 1.
				gap = RNorm * RNorm
			}
			// # ry_sum = np.sum(R * y)
			RY.MulElem(R, Y)
			// # l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
			l21norm = 0.
			for ii = 0; ii < NFeatures; ii++ {
				l21norm += mat.Norm(w.RowView(ii), 2)
			}
			gap += l1reg*l21norm - cons*mat.Sum(RY) + .5*l2reg*(1.+cons*cons)*(wNorm*wNorm)
			if gap < tol {
				// return if we have reached the desired tolerance
				break
			}
		}

	}
	return gap, tol, nIter + 1
}
