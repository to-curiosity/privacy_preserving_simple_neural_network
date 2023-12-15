# privacy_preserving_simple_neural_network
Basic Neural Network application of homomorphic encryption...

///////////////////////------------------------------------------------------------ Background ------------------------------------------------------------///////////////////////
- This project explores the integration of homomorphic encryption into a neural network model, leveraging the MNIST dataset of handwritten digits. The primary aim is to maintain data privacy while ensuring minimal impact on model performance.
- For this project, the chosen dataset is the well-known MNIST collection, which comprises a substantial set of handwritten digits. These digits are each represented by a 28x28 pixel black and white image. To facilitate processing, the dataset was reformatted into a CSV file, where each row corresponds to an individual image. The MNIST dataset encompasses a total of 60,000 images for training and an additional 10,000 for testing purposes. In the CSV format utilized for this project, every row represents a distinct image, with the grayscale intensity of each pixel captured as a column value, ranging from 0 to 255. This format allows for a straightforward and efficient way to handle and process the image data.
- Initially, a user or client encrypts their private data using a private key, and then sends this encrypted data along with their public key to the server. Upon receiving these, the server encrypts the neural network's model weights, which include both the 'Input to Hidden' and 'Hidden to Output' weights.The server then carries out matrix multiplication between the encrypted client data and the now-encrypted model weights (Input to Hidden layer). The result of this computation is subsequently sent back to the client. At this stage, the client decrypts the data, applies an activation function, re-encrypts the data, and sends it back to the server. It's important to note that the client is tasked with applying the activation function, primarily to reduce the computational load and complexity on the server. Incorporating homomorphic approximations of activation functions would significantly increase computational demands and complexity.Once the server receives the data again, it performs another round of matrix multiplication, this time between the encrypted client data and the encrypted model weights (Hidden to Output layer). This processed data is then sent back to the client for the final decryption and application of the activation function. The completion of these steps results in the final output, which is the prediction made by the neural network model. This intricate process underscores the balance between maintaining data privacy and ensuring computational efficiency in the system.


![image](https://github.com/to-curiosity/privacy_preserving_simple_neural_network/assets/67398331/07391322-5693-40b1-a642-ddd4e5f62c04)


///////////////////////------------------------------------------------------------ Run code ------------------------------------------------------------///////////////////////
You will need all the files in the root folder
- To run
- 1.) slect a row number in the train.csv file to pridict, or create your own 28x28 image with a written number. Make sure to convert to a csv
- -> data_to_test,_ := ReadSpecificRow("mnist_test.csv", 497)   //line 432 in the main.go file
- 2.) run the following command: go run main.go homomorphic_encryption_stuffs.go


youtube video for additional info: https://youtu.be/9Ne0RSymm-g?si=PusZL3tvv969AhRv
