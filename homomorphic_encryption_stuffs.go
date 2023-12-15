package main

import (
	"fmt"
	"time"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/bfv"
)

const offset = 10000 // Adjust based on expected range of values

func MulMatrix(matrix1, matrix2 [][]int64) [][]int64 {
	// Check if the matrices have compatible dimensions.
	if len(matrix1[0]) != len(matrix2) {
		fmt.Println("The matrices must have the same number of columns in order to be multiplied.")
		return nil
	}

	// Create a new matrix to store the result.
	result := make([][]int64, len(matrix1))
	for i := range result {
		result[i] = make([]int64, len(matrix2[0]))
	}

	// Perform the matrix multiplication.
	for i := range result {
		for j := range result[0] {
			for k := range matrix1[0] {
				result[i][j] += matrix1[i][k] * matrix2[k][j]
			}
		}
	}

	return result
}

func matrixToSlices(matrix [][]int64, T uint64) [][]uint64 {
    n := len(matrix)
    m := len(matrix[0])

    slices := make([][]uint64, n*m)

    for i := 0; i < n; i++ {
        for j := 0; j < m; j++ {
            val := matrix[i][j]
            if val < 0 {
                val += int64(T)
            }
            slices[i*m+j] = []uint64{uint64(val)}
        }
    }

    return slices
}

func encrypted_mult(matrixA, matrixB [][]int64, T uint64) [][]int64 {
	// Create and initialize MatrixA
	var start time.Time

	// Print matrices then calculate matrix mult
	fmt.Println("matrix A:", matrixA)
	fmt.Println("matrix B:", matrixB)
	start = time.Now()
	matrix_mult_result := MulMatrix(matrixA, matrixB)
	fmt.Println("A*B =:", matrix_mult_result)
	fmt.Println()
	duration := time.Since(start)
	ms := duration.Seconds() * 1000
	fmt.Printf("Regular Calc Done in %.6f ms\n", ms)
	fmt.Println()

	// homomorphic encryption section
	// BFV parameters (128 bit security) with plaintext modulus 65929217
	paramDef := bfv.PN13QP218
	paramDef.T = 0x3ee0001 // 0x3ee0001 hex --> 65929217 decimal

	params, err := bfv.NewParametersFromLiteral(paramDef)
	if err != nil {
		panic(err)
	}

	// setup scheme variables
	encoder := bfv.NewEncoder(params)
	kgen := bfv.NewKeyGenerator(params)
	firstSk, firstPk := kgen.GenKeyPair()
	decryptor := bfv.NewDecryptor(params, firstSk)
	encryptor_firstPk := bfv.NewEncryptor(params, firstPk)
	encryptor_firstSk := bfv.NewEncryptor(params, firstSk)
	evaluator := bfv.NewEvaluator(params, rlwe.EvaluationKey{})

	fmt.Println("================================================================")
	fmt.Println("	  Homomorphic computations on batched integers")
	fmt.Println("================================================================")
	fmt.Println()
	fmt.Printf("Parameters : N=%d, T=%d, Q = %d bits, sigma = %f \n", 1<<params.LogN(), params.T(), params.LogQP(), params.Sigma())
	fmt.Println()

	// Get # of rows in matrix
	numRowsA := len(matrixA)
	numRowsB := len(matrixB)
	// Get # of columns in matrix
	numColsA := len(matrixA[0])
	numColsB := len(matrixB[0])

	size := 0
	if numColsA == numRowsB {
		size = numColsA
	} else {
		fmt.Println("Matrices don't match")
	}

	start = time.Now()

	// Flatten matrices array
	resultMatrixA := matrixToSlices(matrixA, T)
    resultMatrixB := matrixToSlices(matrixB, T)

	// Create plaintext, convert array to polynomials in R_t
	plain_matrixA := make([]*rlwe.Plaintext, numRowsA*numColsA)
	for i := range plain_matrixA {
		plain_matrixA[i] = bfv.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(resultMatrixA[i], plain_matrixA[i])
	}
	plain_matrixB := make([]*rlwe.Plaintext, numRowsB*numColsB)
	for i := range plain_matrixB {
		plain_matrixB[i] = bfv.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(resultMatrixB[i], plain_matrixB[i])
	}

	// Encrypt plaintext to create ciphertext
	ACiphertext := make([]*rlwe.Ciphertext, len(plain_matrixA))
	for i := range ACiphertext {
		ACiphertext[i] = encryptor_firstSk.EncryptNew(plain_matrixA[i])
	}
	BCiphertext := make([]*rlwe.Ciphertext, len(plain_matrixB))
	for i := range BCiphertext {
		BCiphertext[i] = encryptor_firstPk.EncryptNew(plain_matrixB[i])
	}

	// Define the result matrix (slice of slices)
	start = time.Now()
	resultCiphertextMatrix := make([]*rlwe.Ciphertext, numRowsA*numColsB)
	val0 := make([]*rlwe.Ciphertext, size)
	val1 := make([]*rlwe.Ciphertext, size)
	acum := 0
	for i := 0; i < numColsB; i++ {
		for j := 0; j < numRowsA*numColsA; j += numColsA {
			for k := 0; k < numRowsB; k++ {
				val0[k] = evaluator.MulNew(ACiphertext[j+k], BCiphertext[i+(k*numColsB)])
			}
			shallowCopy := val0[0].CopyNew()
			for ii := 0; ii < (len(val0) - 1); ii++ {
				val1[ii] = evaluator.AddNew(shallowCopy, val0[ii+1])
				shallowCopy = val1[ii]
			}

			resultCiphertextMatrix[acum] = shallowCopy
			acum += 1
		}
	}

	// Construct output matrix
	output_matrix := make([][]*rlwe.Ciphertext, numRowsA)
	for i := 0; i < numRowsA; i++ {
		output_matrix[i] = make([]*rlwe.Ciphertext, numColsB)
		for j := 0; j < numColsB; j++ {
			output_matrix[i][j] = resultCiphertextMatrix[i+j*numRowsA]
		}
	}

	fmt.Printf("Homomorphic calc Done in %d ms\n", time.Since(start).Milliseconds())

    result := make([][]int64, numRowsA)
    for i := 0; i < numRowsA; i++ {
        result[i] = make([]int64, numColsB)
        for j := 0; j < numColsB; j++ {
            decryptedResult := decryptor.DecryptNew(output_matrix[i][j])
            decodedResult := encoder.DecodeUintNew(decryptedResult)
            val := int64(decodedResult[0])
            if val > int64(T)/2 {
                val -= int64(T)
            }
            result[i][j] = val
        }
    }
	return result
}

/*
func main() {
	// Example matrices with int64 values
	matrixA := [][]int64{
		{1, -2, 0, 5},
		{3, 4, -2, -1},
		{-3, 1, 3, -9},
		{10, -8, 4, 6},
	}
	matrixB := [][]int64{
		{5, 6, 1, -1},
		{-7, 8, 6, 9},
		{5, -2, 2, -3},
		{-10, 3, 4, 7},
	}

	// Define your plaintext modulus T based on your encryption parameters
	// For this example, I'll use a placeholder value. Replace it with your actual T value.
	const T uint64 = 0x3ee0001 // This should match the T value used in your encryption parameters

	// Call the encrypted_mult function
	result := encrypted_mult(matrixA, matrixB, T)

	// Print the result
	fmt.Println("Encrypted multiplication result:")
	for _, row := range result {
		fmt.Println(row)
	}
}
*/
