import numpy as np
# np.random.seed(0)

#input: dataset X, number of iterations L
#output: rank 1 lc-PCA solution to Qprop, binary vector solution bprop

def l1pca(X, L):
    #the matlab version takes an array that is 2x34 and then we set N to 34.

    #may need to change what i set n to
    N = np.size(X, 1)

    max_iterations = 1000
    iterations = 0

    delta = np.zeros(N, dtype = int)

    obj_val = 0
    bopt = np.zeros(N, dtype = int)


    for i in range(0, L):
        # creates a bitvector of size N of bits -1 or 1
        b = np.random.randint(2, size = N)
        b[b == 0] = -1

        #Iterations for bit flipping to converge
        for iterations in range(0, max_iterations):
            ##print()
            print()

            #loop over each N bits in B
            for k in range(0, N):
                bk = np.delete(b, k)
                Xk = np.delete(X, k, axis = 1)
                print("original b: {} \n new bi: {}".format(b, bk))
                print("original X: {} \n new Xi: {}".format(X, Xk))

                firstHalfMultiply = (-4 * b[k] * np.transpose(X[:,k]))
                print("multiplying {} by {} and -4 \n result: {}".format(b[k],  np.transpose(X[:,k]), firstHalfMultiply))
                print("first half {}, first half shape: ".format(firstHalfMultiply, firstHalfMultiply.shape))

                secondHalfMultiply = np.matmul(Xk, bk)
                print("Xi shape: {}, bi shape: {}".format(Xk.shape, bk.shape))
                print("second half, (Xi * bi): ", secondHalfMultiply)

                delta[k] = np.matmul(firstHalfMultiply, secondHalfMultiply)
                print("After multiplying Xk* bk * previous result: {}".format(delta[k]))
                print()


            maxBindex = np.argmax(delta)
            print("Current b = {}".format(b))
            print("Delta: {} \nargmax in the delta: {}".format(delta, maxBindex))
            if delta[maxBindex] > 0:
                print("flipping bit {}".format(maxBindex))
                b[maxBindex] *= -1
                print("New b = {}".format(b))
            else:
                print("no improvement, breaking now.\nCurrent b = {}".format(b))
                break

        # ord = inf uses the norm for matrix as max(sum(abs(x), axis=1))
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html

        temporaryNorm = np.linalg.norm(np.matmul(X, b), 2)

        print("temporary norm of the given b: {}".format(temporaryNorm))
        print("temporary norm of the given b: {} comparing against old norm: {}\nold bopt: {} new potential bopt: {}".format(temporaryNorm, obj_val, bopt, b))

        if temporaryNorm > obj_val:
            print("found new bopt.\nold objval: {} new objval: {}\nold bopt: {} new bopt: {}".format(obj_val, temporaryNorm, bopt, b))
            obj_val = temporaryNorm
            bopt = b
            i_best = i
            # print("new i best: {}".format(i_best))
            print("going to initizlization ", i+1)
        #else:
            print("didnt beat old values, going to initizlization {}".format(i+1))


    Qmult = np.matmul(X, bopt)
    Qprop = Qmult / np.linalg.norm(Qmult, 2)
    Bprop = bopt
    # return Qprop, Bprop, iterations, i_best
    # print(Qprop)
    return Qprop




X = np.array([[1, 2, 3], [4, 5, 6]])
L = 4
print(l1pca(X, L))

# mu, sigma = 0, 1
# normaldist = np.random.normal(mu, sigma, (2, 6))
#
# myInitializations = 3
#
# #print("shape of normal dist: {}".format(normaldist.shape))
# print(l1pca(normaldist, myInitializations))
