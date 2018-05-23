package linearModel

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// coordinate descent algorithm for Elastic-Net
// v https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/linear_model/cd_fast.pyx
func enetCoordinateDescent(w *mat.Dense, alpha, beta float64, X, Y *mat.Dense, maxIter int, tol float64, rng rand.Rand, random, positive bool) (gap, tolyy float64, nIter int) {
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()
	R := mat.NewDense(NSamples, NOutputs, nil)
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
	XW := &mat.Dense{}
	XW.Mul(X, w)
	R.Copy(Y)
	R.Sub(R, XW)
	// # tol = tol * linalg.norm(Y, ord='fro') ** 2
	Y2 := &mat.Dense{}
	Y2.MulElem(Y, Y)
	tol *= mat.Sum(Y2)
	wii := make([]float64, NOutputs)
	tmp := make([]float64, NOutputs)
	for nIter = 0; nIter < maxIter; nIter++ {
		//wMax, dwMax := 0., 0.
		var ii int
		for fIter := 0; fIter < NFeatures; fIter++ {
			if random {
				ii = rng.Intn(NFeatures)
			} else {
				ii = fIter
			}
			if normColsX[ii] == 0. {
				continue
			}
			// # w_ii = W[:, ii] # Store previous value
			mat.Row(wii, ii, w)
			// # if np.sum(w_ii ** 2) != 0.0:  # can do better
			if mat.Norm(mat.NewVecDense(len(wii), wii), 1) != 0. {
				//# R += np.dot(X[:, ii][:, None], w_ii[None, :]) # rank 1 update
				XW.Mul(X.Slice(0, NSamples, ii, ii+1), mat.NewDense(1, NOutputs, wii))
				R.Add(R, XW)
			}
			// # tmp = np.dot(X[:, ii][None, :], R).ravel()
			mat.NewDense(1, NOutputs, tmp).Mul(X.Slice(0, NSamples, ii, ii+1).T(), R)
			// # nn = sqrt(np.sum(tmp ** 2))
			//nn := mat.Norm(mat.NewDense(1, NOutputs, tmp), 2)
			// # W[:, ii] = tmp * fmax(1. - l1_reg / nn, 0) / (norm_cols_X[ii] + l2_reg)
			for o := 0; o < NOutputs; o++ {
				//	w.Set(ii, o, tmp[o]*math.Max(0, 1-l1_reg)/(normColsX[ii])+l2_reg)
			}
		}

	}
	return
}

/*
        # R = y - np.dot(X, w)
        for i in range(n_samples):
            R[i] = y[i] - dot(n_features, &X_data[i], n_samples, w_data, 1)

        # tol *= np.dot(y, y)
        tol *= dot(n_samples, y_data, n_tasks, y_data, n_tasks)

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    axpy(n_samples, w_ii, &X_data[ii * n_samples], 1,
                         R_data, 1)

                # tmp = (X[:,ii]*R).sum()
                tmp = dot(n_samples, &X_data[ii * n_samples], 1, R_data, 1)

                if positive and tmp < 0:
                    w[ii] = 0.0
                else:
                    w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                             / (norm_cols_X[ii] + beta))

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    axpy(n_samples, -w[ii], &X_data[ii * n_samples], 1,
                         R_data, 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if (w_max == 0.0 or
                d_w_max / w_max < d_w_tol or
                n_iter == max_iter - 1):
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion

                # XtA = np.dot(X.T, R) - beta * w
                for i in range(n_features):
                    XtA[i] = dot(n_samples, &X_data[i * n_samples],
                                 1, R_data, 1) - beta * w[i]

                if positive:
                    dual_norm_XtA = max(n_features, XtA_data)
                else:
                    dual_norm_XtA = abs_max(n_features, XtA_data)

                # R_norm2 = np.dot(R, R)
                R_norm2 = dot(n_samples, R_data, 1, R_data, 1)

                # w_norm2 = np.dot(w, w)
                w_norm2 = dot(n_features, w_data, 1, w_data, 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = asum(n_features, w_data, 1)

                # np.dot(R.T, y)
                gap += (alpha * l1_norm
                        - const * dot(n_samples, R_data, 1, y_data, n_tasks)
                        + 0.5 * beta * (1 + const ** 2) * (w_norm2))

                if gap < tol:
                    # return if we reached desired tolerance
                    break
return w, gap, tol, n_iter + 1
*/
