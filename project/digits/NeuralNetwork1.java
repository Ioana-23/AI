package project.digits;

import java.util.Arrays;

public class NeuralNetwork1 {
    private double[][] X;//every row contains the vector of pixels in an image
    private double[][] Y;//contains the labels
    private int inputSize;
    private int hiddenNodes;
    private int outputSize;
    private double[][] weights1;
    private double[][] weights2;
    private double[][] biases1;
    private double[][] biases2;
    private int epochs;
    private int entrySize;
    private double learningRate;
    public NeuralNetwork1(double learningRate, int entrySze,int inputSize, double[][] x, double[][] y, int hiddenNodes, int outputSize, int epochs)
    {
        this.learningRate = learningRate;
        this.entrySize = entrySze;
        this.inputSize = inputSize;
        this.X = x;
        this.Y = y;
        this.hiddenNodes = hiddenNodes;
        this.outputSize = outputSize;
        this.epochs = epochs;
        initialize();
    }
    private void initialize()
    {
        X = MatrixUtils.T(X);
        Y = MatrixUtils.T(Y);
        weights1 = MatrixUtils.random(hiddenNodes, inputSize);
        biases1 = new double[hiddenNodes][entrySize];
        weights2 = MatrixUtils.random(outputSize, hiddenNodes);
        biases2 = new double[outputSize][entrySize];
    }
    public void train()
    {
        for(int i = 0; i < epochs; i++)
        {
            //Forward Prop
            //Layer 1
            double[][] Z1 = MatrixUtils.add(MatrixUtils.dot(weights1, X), biases1);
            double[][] A1 = MatrixUtils.sigmoid(Z1);

            //LAYER 2
            double[][] Z2 = MatrixUtils.add(MatrixUtils.dot(weights2, A1), biases2);
            double[][] A2 = MatrixUtils.sigmoid(Z2);

            double cost = MatrixUtils.cross_entropy(entrySize, Y, A2);
            //costs.getData().add(new XYChart.Data(i, cost));

            // Back Prop
            //LAYER 2
            double[][] dZ2 = MatrixUtils.subtract(A2, Y);
            double[][] dweights2 = MatrixUtils.divide(MatrixUtils.dot(dZ2, MatrixUtils.T(A1)), entrySize);
            double[][] dbiases2 = MatrixUtils.divide(dZ2, entrySize);

            //LAYER 1
            double[][] dZ1 = MatrixUtils.multiply(MatrixUtils.dot(MatrixUtils.T(weights2), dZ2), MatrixUtils.subtract(1.0, MatrixUtils.power(A1, 2)));
            double[][] dweights1 = MatrixUtils.divide(MatrixUtils.dot(dZ1, MatrixUtils.T(X)), entrySize);
            double[][] dbiases1 = MatrixUtils.divide(dZ1, entrySize);

            // G.D
            weights1 = MatrixUtils.subtract(weights1, MatrixUtils.multiply(0.01, dweights1));
            biases1 = MatrixUtils.subtract(biases1, MatrixUtils.multiply(0.01, dbiases1));

            weights2 = MatrixUtils.subtract(weights2, MatrixUtils.multiply(0.01, dweights2));
            biases2 = MatrixUtils.subtract(biases2, MatrixUtils.multiply(0.01, dbiases2));
            if(i%hiddenNodes==0)
            {
                System.out.println("==============");
                System.out.println("Cost = " + cost);
                System.out.println("Predictions = " + Arrays.deepToString(A2));
            }
        }
    }
}
