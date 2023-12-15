# privacy_preserving_simple_neural_network
Basic Neural Network application of homomorphic encryption...

///////////////////////------------------------------------------------------------ Background ------------------------------------------------------------///////////////////////
- This project explores the integration of homomorphic encryption into a neural network model, leveraging the MNIST dataset of handwritten digits. The primary aim is to maintain data privacy while ensuring minimal impact on model performance.
- For this project, the chosen dataset is the well-known MNIST collection, which comprises a substantial set of handwritten digits. These digits are each represented by a 28x28 pixel black and white image. To facilitate processing, the dataset was reformatted into a CSV file, where each row corresponds to an individual image. The MNIST dataset encompasses a total of 60,000 images for training and an additional 10,000 for testing purposes. In the CSV format utilized for this project, every row represents a distinct image, with the grayscale intensity of each pixel captured as a column value, ranging from 0 to 255. This format allows for a straightforward and efficient way to handle and process the image data.

///////////////////////------------------------------------------------------------ Run code ------------------------------------------------------------///////////////////////
You will need all the files in the root folder
- To run
- 1.) slect a row number in the train.csv file to pridict, or create your own 28x28 image with a written number. Make sure to convert to a csv
- -> data_to_test,_ := ReadSpecificRow("mnist_test.csv", 497)   //line 432 in the main.go file
- 2.) run the following command: go run main.go homomorphic_encryption_stuffs.go
