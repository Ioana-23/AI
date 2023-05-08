package project.digits;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class NeuralNet {
    // Notes:
    // The first thing I did is to make the code a bit more dynamic, there was a clear problem with the backpropagation part,
    // but I wasn't sure if this was the only bug in the code. So to make sure I refactored the whole code to make it a bit more readable.
    // I made two other class:
    // 1) StatUtil, which is a static class containing stats functions.
    // 2) Layer, which represent a layer with array on neurons inside them.
    // The last class was useful so to modularize the layers and not hardcode them.
    // The first class was useful to remove some code from the main method and make it less bulky.
    // Next thing I did is to remove some variable from the code and put them in the neuron class(min/maxWeight).
    // And to add some variables in the neurons class.
    // When the code was more readable I went to rewrite the forward propagation (which you got right).
    // I then rewrote the back propagation algorithm and this is where you messed up some stuff.
    // Here is a very good step by step explanation of every part of the backprop algorithm : https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    // One thing to note is that the biases are still not implemented as I didn't have enough time to include them. It isn't a problem
    // for the XOR function, but it might be if you are trying to learn a function requiring some degrees of translation.

    // Variable Declaration

    // Layers
    static Layer[] layers; // My changes
    private static final String readTrainImages ="D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\train-images.idx3-ubyte";
    private static final String readTrainLabels ="D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\train-labels.idx1-ubyte";
    private static final String readTestImages = "D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\t10k-images.idx3-ubyte";
    private static final String readTestLabels = "D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\t10k-labels.idx1-ubyte";
    static double[][] trainImages;
    static double[][] trainLabels;
    static double[][] testImages;
    static double[][] testLabels;
    // Training data
    static TrainingData[] tDataSet; // My changes

    // Main Method
    public static void main(String[] args) throws IOException {
        // My changes
        // Set the Min and Max weight value for all Neurons
        Neuron.setRangeWeight(-1,1);

        // Create the layers
        // Notes: One thing you didn't code right is that neurons in a layer
        // need to have number of weights corresponding to the previous layer
        // which means that the first hidden layer need to have 2 weights per neuron and 6 neurons
        layers = new Layer[3];
        layers[0] = null; // Input Layer 0,2
        layers[1] = new Layer(784,128); // Hidden Layer 2,6
        layers[2] = new Layer(128,10); // Output Layer 6,1

        // Create the training data
        CreateTrainingData();

        System.out.println("============");
        System.out.println("Output before training");
        System.out.println("============");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }

        train(100, 0.05f);

        System.out.println("============");
        System.out.println("Output after training");
        System.out.println("============");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }
    }

    public static void CreateTrainingData() throws IOException {
        DataInputStream in = new DataInputStream(new FileInputStream(readTrainImages));
        int magicNumber = in.readInt();
        int numImages = in.readInt();
        int numRows = in.readInt();
        int numCols = in.readInt();
        trainImages = new double[200][numRows*numCols];
        for(int i = 0; i < 200; i++)
        {
            for(int j = 0; j < numCols * numRows; j++)
            {
                trainImages[i][j] = in.readUnsignedByte()/255d;
            }
        }
        in.close();
        in = new DataInputStream(new FileInputStream(readTrainLabels));
        magicNumber = in.readInt();
        int numLabels = in.readInt();
        trainLabels = new double[200][10];
        for(int i = 0; i < 200; i++)
        {
            for(int j = 0; j < 10; j++)
            {
                trainLabels[i][j] = 0;
            }
            trainLabels[i][in.readUnsignedByte()] = 1;
        }
        in.close();

        in = new DataInputStream(new FileInputStream(readTestImages));
        magicNumber = in.readInt();
        numImages = in.readInt();
        numRows = in.readInt();
        numCols = in.readInt();
        testImages = new double[200][numRows*numCols];
        for(int i = 0; i < 200; i++)
        {
            for(int j = 0; j < numCols * numRows; j++)
            {
                testImages[i][j] = in.readUnsignedByte();
            }
        }
        in.close();
        in = new DataInputStream(new FileInputStream(readTestLabels));
        magicNumber = in.readInt();
        numLabels = in.readInt();
        testLabels = new double[200][10];
        for(int i = 0; i < 200; i++)
        {
            for(int j = 0; j < 10; j++)
            {
                testLabels[i][j] = 0;
            }
            testLabels[i][in.readUnsignedByte()] = 1;
        }
        in.close();
        tDataSet = new TrainingData[200];
        for(int i = 0; i < 200; i++)
        {
            double[] input = trainImages[i];
            double[] expectedOutput = trainLabels[i];
            tDataSet[i] = new TrainingData(input, expectedOutput);
        }
    }

    public static void forward(double[] inputs) {
        // First bring the inputs into the input layer layers[0]
        layers[0] = new Layer(inputs);

        for(int i = 1; i < layers.length; i++) {
            for(int j = 0; j < layers[i].neurons.length; j++) {
                float sum = 0;
                for(int k = 0; k < layers[i-1].neurons.length; k++) {
                    sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
                }
                //sum += layers[i].neurons[j].bias; // TODO add in the bias
                layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
            }
        }
    }

    // This part is heavily inspired from the website in the first note.
    // The idea is that you calculate a gradient and cache the updated weights in the neurons.
    // When ALL the neurons new weight have been calculated we refresh the neurons.
    // Meaning we do the following:
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public static void backward(float learning_rate,TrainingData tData) {

        int number_layers = layers.length;
        int out_index = number_layers-1;

        // Update the output layers
        // For each output
        for(int i = 0; i < layers[out_index].neurons.length; i++) {
            // and for each of their weights
            double output = layers[out_index].neurons[i].value;
            double target = tData.expectedOutput[i];
            double derivative = output-target;
            double delta = derivative*(output*(1-output));
            layers[out_index].neurons[i].gradient = delta;
            for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) {
                double previous_output = layers[out_index-1].neurons[j].value;
                double error = delta*previous_output;
                layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
            }
        }

        //Update all the subsequent hidden layers
        for(int i = out_index-1; i > 0; i--) {
            // For all neurons in that layers
            for(int j = 0; j < layers[i].neurons.length; j++) {
                double output = layers[i].neurons[j].value;
                double gradient_sum = sumGradient(j,i+1);
                double delta = (gradient_sum)*(output*(1-output));
                layers[i].neurons[j].gradient = delta;
                // And for all their weights
                for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
                    double previous_output = layers[i-1].neurons[k].value;
                    double error = delta*previous_output;
                    layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
                }
            }
        }

        // Here we do another pass where we update all the weights
        for(int i = 0; i< layers.length;i++) {
            for(int j = 0; j < layers[i].neurons.length;j++) {
                layers[i].neurons[j].update_weight();
            }
        }

    }

    // This function sums up all the gradient connecting a given neuron in a given layer
    public static float sumGradient(int n_index,int l_index) {
        float gradient_sum = 0;
        Layer current_layer = layers[l_index];
        for(int i = 0; i < current_layer.neurons.length; i++) {
            Neuron current_neuron = current_layer.neurons[i];
            gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
        }
        return gradient_sum;
    }


    // This function is used to train being forward and backward.
    public static void train(int training_iterations,float learning_rate) {
        for(int i = 0; i < training_iterations; i++) {
            for(int j = 0; j < tDataSet.length; j++) {
                forward(tDataSet[j].data);
                backward(learning_rate,tDataSet[j]);
            }
        }
    }
}
