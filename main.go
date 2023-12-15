package main

import (
	"encoding/csv"
	"reflect"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
	"encoding/gob"
)

type NeuralNetwork struct {
	inputSize  int
	hiddenSize int
	outputSize int
	weightsIH  [][]float64
	weightsHO  [][]float64
	rate       float64
}

func transpose(slice []float64) [][]float64 {
    transposed := make([][]float64, len(slice))
    for i, val := range slice {
        transposed[i] = []float64{val}
    }
    return transposed
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}


func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func (nn *NeuralNetwork) Init(input, hidden, output int, rate float64) {
    nn.inputSize = input
    nn.hiddenSize = hidden
    nn.outputSize = output
    nn.rate = rate
}

func (nn *NeuralNetwork) InitializeWeights() {
    nn.weightsIH = make([][]float64, nn.hiddenSize)
    for i := range nn.weightsIH {
        nn.weightsIH[i] = make([]float64, nn.inputSize)
        for j := range nn.weightsIH[i] {
            nn.weightsIH[i][j] = rand.Float64() - 0.5
        }
    }

    nn.weightsHO = make([][]float64, nn.outputSize)
    for i := range nn.weightsHO {
        nn.weightsHO[i] = make([]float64, nn.hiddenSize)
        for j := range nn.weightsHO[i] {
            nn.weightsHO[i][j] = rand.Float64() - 0.5
        }
    }
}

func (nn *NeuralNetwork) SaveWeights(filepath string) error {
    file, err := os.Create(filepath)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := gob.NewEncoder(file)
    err = encoder.Encode(nn.weightsIH)
    if err != nil {
        return err
    }
    err = encoder.Encode(nn.weightsHO)
    return err
}

func (nn *NeuralNetwork) LoadWeights(filepath string) error {
    file, err := os.Open(filepath)
    if err != nil {
        return err
    }
    defer file.Close()
    decoder := gob.NewDecoder(file)
    err = decoder.Decode(&nn.weightsIH)
    if err != nil {
        return err
    }
    err = decoder.Decode(&nn.weightsHO)
    if err != nil {
        return err
    }
    // Count the number of weights in weightsIH and weightsHO
    numWeightsIH := 0
    for _, row := range nn.weightsIH {
        numWeightsIH += len(row)
    }

    numWeightsHO := 0
    for _, row := range nn.weightsHO {
        numWeightsHO += len(row)
    }

	for i := range nn.weightsHO {
		for j := range nn.weightsHO[i] {
			nn.weightsHO[i][j] = math.Round(nn.weightsHO[i][j] * 10)
		}
	}
	

	for i := range nn.weightsIH {
		for j := range nn.weightsIH[i] {
			nn.weightsIH[i][j] = math.Round(nn.weightsIH[i][j] * 10)
		}
	}
	
	//fmt.Println(nn.weightsHO)

    fmt.Printf("Number of weights in weightsIH: %d\n", numWeightsIH)
    fmt.Printf("Number of weights in weightsHO: %d\n", numWeightsHO)
    fmt.Printf("Total number of weights: %d\n", numWeightsIH + numWeightsHO)

    return nil
}

func (nn *NeuralNetwork) Forward(input []float64) ([]float64, []float64) {
	///This is where we can insert the homomorphic encyption 
	hiddenOutputs := make([]float64, nn.hiddenSize)
	for i := range nn.weightsIH {
		for j, val := range input {
			hiddenOutputs[i] += val * nn.weightsIH[i][j]   
		}
		hiddenOutputs[i] = sigmoid(hiddenOutputs[i])
		//fmt.Println("HiddenOutputs length: ", len(hiddenOutputs))
	}
	finalOutputs := make([]float64, nn.outputSize)
	for i := range nn.weightsHO {
		for j, val := range hiddenOutputs {
			finalOutputs[i] += val * nn.weightsHO[i][j]  
		}
		finalOutputs[i] = sigmoid(finalOutputs[i])
		//fmt.Println("FinalOutputs length: ", len(finalOutputs))
	}

	//fmt.Println("Type input: ", reflect.TypeOf(input))
	//fmt.Println("Type nn.weightsIH: ", reflect.TypeOf(nn.weightsIH))
	//fmt.Println("Type nn.weightsHO: ", reflect.TypeOf(nn.weightsHO))
	return hiddenOutputs, finalOutputs
}

func ConvertFloat64MatrixToInt64(floatMatrix [][]float64) [][]int64 {
    intMatrix := make([][]int64, len(floatMatrix))

    for i, row := range floatMatrix {
        intMatrix[i] = make([]int64, len(row))
        for j, val := range row {
            // Converting float64 to int64
            intMatrix[i][j] = int64(val)
        }
    }

    return intMatrix
}

func ConvertInt64ToFloat64Matrix(intMatrix [][]int64) [][]float64 {
	floatMatrix := make([][]float64, len(intMatrix))

	for i, row := range intMatrix {
		floatMatrix[i] = make([]float64, len(row))
		for j, val := range row {
			floatMatrix[i][j] = float64(val)
		}
	}

	return floatMatrix
}

func (nn *NeuralNetwork) Forward_enc(input, weightsIH, weightsHO [][]float64) {
	const T uint64 = 0x3ee0001
	//result := encrypted_mult(matrixA, matrixB, T)
	
	//input:     not normalized yet
	//weightsIH: Already scalled and absolute valued
	//weightsHO: Already scalled and absolute valued
	
	input_1 := ConvertFloat64MatrixToInt64(input)
	weights_IH := ConvertFloat64MatrixToInt64(weightsIH)
	hiddenOutputs := encrypted_mult(weights_IH,input_1,T)
	print("size of hidden outputs: ")
	fmt.Println("Encypted hiddenOutputs Result: ",hiddenOutputs)

	hiddenOutputs_array := make([][]float64, len(hiddenOutputs))
	HO_arr := ConvertInt64ToFloat64Matrix(hiddenOutputs)

	for i, row := range HO_arr {
		hiddenOutputs_array[i] = make([]float64, len(row))
		for j, val := range row {
			hiddenOutputs_array[i][j] = sigmoid(val)
		}
	}
	fmt.Println("After Sigmoid Result: ",hiddenOutputs)

	weights_HO := ConvertFloat64MatrixToInt64(weightsHO)
	hiddenOutputs_array2 := ConvertFloat64MatrixToInt64(hiddenOutputs_array)
	finalOutputs := encrypted_mult(weights_HO,hiddenOutputs_array2, T)
	fmt.Println("Encypted finalOutputs Result: ",finalOutputs)

	finalOutputs_array := make([][]float64, len(finalOutputs))
	FO_arr := ConvertInt64ToFloat64Matrix(finalOutputs)

	for i, row := range FO_arr {
		finalOutputs_array[i] = make([]float64, len(row))
		for j, val := range row {
			finalOutputs_array[i][j] = sigmoid(val)
		}
	}
	//fmt.Println("After Sigmoid Result: ",finalOutputs_array)
	maxValue := 0.0
	prediction := 0

	// Flatten the 2D array and find the max value
	for i, row := range finalOutputs_array {
		for j, value := range row {
			if value > maxValue {
				maxValue = value
				prediction = j + i*len(row) // calculate the position in a flattened array
			}
		}
	}

	fmt.Printf("Prediction: %d with value: %f\n", prediction, maxValue)

}

func (nn *NeuralNetwork) Backpropagate(input, hiddenOutputs, finalOutputs, targets []float64) {
	outputErrors := make([]float64, nn.outputSize)
	for i, finalOutput := range finalOutputs {
		outputErrors[i] = targets[i] - finalOutput
	}
	hiddenErrors := make([]float64, nn.hiddenSize)
	for i := range nn.weightsHO {
		for j := range nn.weightsHO[i] {
			nn.weightsHO[i][j] += nn.rate * outputErrors[i] * sigmoidDerivative(finalOutputs[i]) * hiddenOutputs[j]
			hiddenErrors[j] += outputErrors[i] * nn.weightsHO[i][j]
		}
	}
	for i := range nn.weightsIH {
		for j := range nn.weightsIH[i] {
			nn.weightsIH[i][j] += nn.rate * hiddenErrors[i] * sigmoidDerivative(hiddenOutputs[i]) * input[j]
		}
	}
}

// Learning rate scheduling method
func (nn *NeuralNetwork) UpdateLearningRate(epoch, totalEpochs int) {
    initialLearningRate := 0.0955 // Set your initial learning rate
    decayRate := 0.95             // Decay rate per epoch
    nn.rate = initialLearningRate * math.Pow(decayRate, float64(epoch)/float64(totalEpochs))
}

func loadCSVData(filename string) ([][]float64, [][]float64) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputData := make([][]float64, len(records))
	targetData := make([][]float64, len(records))

	for i, record := range records {
		input := make([]float64, len(record)-1)
		target := make([]float64, 10) // Assuming 10 classes for MNIST

		for j := 0; j < len(record)-1; j++ {
			val, err := strconv.ParseFloat(record[j+1], 64)    //skip the first column of file in each row and starts reading from the second column.
			if err != nil {
				log.Fatal(err)
			}
			input[j] = val / 255.0 // Normalize pixel values to the range [0, 1]
		}

		label, err := strconv.Atoi(record[0])
		if err != nil {
			log.Fatal(err)
		}
		target[label] = 1.0

		inputData[i] = input
		targetData[i] = target
	}

	return inputData, targetData
}

func ReadSpecificRow(filePath string, rowNum int) ([]float64, error) {
	// ReadSpecificRow reads a specific row from a CSV file, skipping the first column
    // Open the file
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    // Create a new CSV reader
    reader := csv.NewReader(file)

    // Read all rows
    rows, err := reader.ReadAll()
    if err != nil {
        return nil, err
    }

    // Check if the specified row number is valid
    if rowNum < 0 || rowNum >= len(rows) {
        return nil, fmt.Errorf("row number out of range")
    }

    // Get the specific row and convert it to []float64
    stringRow := rows[rowNum][1:] // Skipping the first column
    floatRow := make([]float64, len(stringRow))
    for i, strValue := range stringRow {
        floatValue, err := strconv.ParseFloat(strValue, 64)
        if err != nil {
            return nil, fmt.Errorf("error converting string to float: %v", err)
        }
        floatRow[i] = floatValue
    }

    return floatRow, nil
}

func main() {
	var start time.Time
	start = time.Now()

    rand.Seed(time.Now().UnixNano())

    // Initialize neural network for MNIST without initializing weights
    nn := NeuralNetwork{}
    nn.Init(784, 10, 10, 0.0955) // 784 input nodes (28x28 pixels), 10 hidden nodes, 10 output nodes, [784, 20, 10, 0.0955]  /hidden was 300

    // Define paths for saving or loading weights
    weightsFilePath := "nn_weights.gob"

    // Check if pre-trained weights exist
    if _, err := os.Stat(weightsFilePath); os.IsNotExist(err) {
        // Pre-trained weights do not exist, so initialize and train the network
        nn.InitializeWeights()

        // Load training data from CSV  file
        trainInput, trainTargets := loadCSVData("mnist_train.csv")

		totalEpochs := 100 // Set the total number of epochs

        // Training with learning rate scheduling
        for epoch := 0; epoch < totalEpochs; epoch++ {
			fmt.Println("Epoch: ",epoch)
            nn.UpdateLearningRate(epoch, totalEpochs) // Update the learning rate

            for i := 0; i < len(trainInput); i++ {
                hiddenOutput, output := nn.Forward(trainInput[i])
                nn.Backpropagate(trainInput[i], hiddenOutput, output, trainTargets[i])
            }

        }

        // Save the trained weights
        if err := nn.SaveWeights(weightsFilePath); err != nil {
            log.Fatalf("Failed to save weights: %v", err)
        }
    } else {
        // Load pre-trained weights
        if err := nn.LoadWeights(weightsFilePath); err != nil {
            log.Fatalf("Failed to load weights: %v", err)
        }
    }

    // Load testing data from CSV file
    testInput, testTargets := loadCSVData("mnist_test.csv")
	//fmt.Println(testInput[50])
	correctPredictions := 0
	for i := 0; i < len(testInput); i++ {
		_, output := nn.Forward(testInput[i])

		// Find the index of the largest value in the output
		maxIndex := 0
		maxValue := output[0]
		for j, value := range output {
			if value > maxValue {
				maxValue = value
				maxIndex = j
			}
		}

		// Check if the prediction matches the actual class
		if testTargets[i][maxIndex] == 1 {
			correctPredictions++
		}
	}

	// fmt.Println("Size of input: ",len(input)) // input nodes (28x28=784 pixels), 9 hidden nodes, 10 output
	fmt.Println("---------------------------------------------------------------------------------------------")
	fmt.Println("Size of input: ",len(testInput[0]))
	//fmt.Println("Input: ",testInput[0])
	fmt.Println("---------------------------------------------------------------------------------------------")
	fmt.Println("Size of input to hidden weights: ",len(nn.weightsIH[1]))
	//fmt.Println("Input to hidden weights: ",nn.weightsIH[1])
	fmt.Println("---------------------------------------------------------------------------------------------")
	fmt.Println("Size of hidden to output weights: ",len(nn.weightsHO))
	//fmt.Println("Hidden to output weights: ",nn.weightsHO)
	fmt.Println("---------------------------------------------------------------------------------------------")

	// Calculate accuracy
	accuracy := float64(correctPredictions) / float64(len(testInput)) * 100
	fmt.Printf("Model Accuracy: %f%%\n", accuracy)
	fmt.Println("Type: ", reflect.TypeOf(nn.weightsHO))

	// -------------------------------- Testing ----------------------------------------------------------------
	data_to_test,_ := ReadSpecificRow("mnist_test.csv", 497)  // test the model on a value in mnist_test.csv, remember csv starts counting from 1

	test_input_prime := transpose(data_to_test)  // size: 784 x 1
	nn.Forward_enc(test_input_prime,nn.weightsIH,nn.weightsHO)
	duration := time.Since(start)
	ms := duration.Seconds() * 1000
	fmt.Printf("HE NN Calc Done in %.6f ms\n", ms)
}




