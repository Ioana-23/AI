package project.digits;

import java.util.List;
import java.util.Random;

public class NeuralNetworkDigits {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] hiddenWeights;
    private double[][] outputWeights;
    private double[] biases1;
    private double[] biases2;
    private double learningRate = 0.01;
    public NeuralNetworkDigits(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenWeights = new double[inputSize][hiddenSize];
        this.outputWeights = new double[hiddenSize][outputSize];
        this.biases1 = new double[hiddenSize];
        this.biases2 = new double[outputSize];
        initializeWeights(hiddenWeights);
        initializeWeights(outputWeights);
    }
    private void initializeWeights(double[][] weights)
    {
        Random random = new Random();
        for(int i = 0; i < weights.length; i++)
        {
            for(int j = 0; j < weights[0].length; j++)
            {
                weights[i][j] = random.nextGaussian() * 0.1;
            }
        }
    }
    private double[][] multiply(double[][] a, double[][] b)
    {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if(n1!=m2)
        {
            throw new IllegalArgumentException("Matrices cannot be multiplied!");
        }
        double[][] result = new double[m1][m2];
        for(int i = 0; i < m1; i++)
        {
            for(int j = 0; j < n2; j++)
            {
                double sum = 0;
                for(int k = 0; k < n1; k++)
                {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    private double[][] transpose(double[][] a)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[n][m];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }
    private double[][] substract(double[][] a, double[][] b)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }
    private double[][] multiply(double[][] a, double b)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                result[i][j] = a[i][j] * b;
            }
        }
        return result;
    }
    private double[][] sigmoid(double[][] a)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                result[i][j] = 1/(1+Math.exp(-a[i][j]));
            }
        }
        return result;
    }
    private double[][] sigmoidDerivative(double[][] a)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                double sigmoid = 1 / (1 + Math.exp(-a[i][j]));
                result[i][j] = sigmoid * (1 - sigmoid);
            }
        }
        return result;
    }
    private double[][] flattenImages(List<int[][]> images)
    {
        int numImages = images.size();
        int numRows = images.get(0).length;
        int numCols = images.get(0)[0].length;
        double[][] flattenedImages = new double[numImages][numRows * numCols];
        for(int i = 0; i < numImages; i++)
        {
            double[] flattenedImage = new double[numRows*numCols];
            for(int row = 0; row < numRows; row++)
            {
                for(int col = 0; col < numCols; col++)
                {
                    flattenedImage[row * numCols + col] = images.get(i)[row][col];
                }
            }
            flattenedImages[i] = flattenedImage;
        }
        return flattenedImages;
    }
    private double[][] add(double[][] a, double[][] b)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    private double[] addBias(double[] a)
    {
        int m = a.length;
        double[] result = new double[m+1];
        result[0] = 1;
        for(int i = 0; i < m; i++)
        {
            result[i+1] = a[i];
        }
        return result;
    }
    private double[][] addBias(double[][] a)
    {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n+1];
        for(int i = 0; i < m; i++)
        {
            result[i][0] = 1;
            for(int j = 1; j<=n; j++)
            {
                result[i][j] = a[i][j-1];
            }
        }
        return result;
    }
    public void train(double[][] X, int[] y, int epochs, double learningRate)
    {
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            double loss = 0;
            for(int i = 0; i < X.length; i++)
            {
                double[][] layer1 = sigmoid(multiply(new double[][]{X[i]}, hiddenWeights));
                double[][] output = sigmoid(multiply(layer1, outputWeights));
                double[][] outputError = new double[i][outputSize];
                outputError[0][y[i]] = output[0][y[i]] - 1;
                double[][] layer1Error = multiply(outputError, transpose(outputWeights));
                double[][] layer1Gradient = multiply(multiply(layer1Error, sigmoidDerivative(layer1)), transpose(new double[][]{X[i]}));
                double[][] outputGradient = multiply(transpose(layer1), outputError);
                hiddenWeights = substract(hiddenWeights, multiply(layer1Gradient, learningRate));
                outputWeights = substract(outputWeights, multiply(outputGradient, learningRate));
            }
        }
    }
    public int predict(double[] x)
    {
        double[][] layer1 = sigmoid(multiply(new double[][]{x}, hiddenWeights));
        double[][] output = sigmoid(multiply(layer1, outputWeights));
        int prediction = 0;
        double maxScore = 0;
        for(int i = 0; i < outputSize; i++)
        {
            if(output[0][i] > maxScore)
            {
                prediction = i;
                maxScore = output[0][i];
            }
        }
        return prediction;
    }
    /*private double[] sigmoid(double[] x)
    {
        double[] result
    }
    public void train(double[][] inputs, double[] labels)
    {
        double[] layer1Outputs = sigmoid()
        *//*double[] hiddenOutputs = new double[hiddenSize];
        double[] outputOutputs = new double[outputSize];
        dot(inputs, hiddenWeights, hiddenOutputs);
        relu(hiddenOutputs);
        dot(hiddenOutputs, outputWeights, outputOutputs);
        softmax(outputOutputs);

        double[] outputGradients = new double[outputSize];
        double[] hiddenGradients = new double[hiddenSize];
        for(int i = 0; i < outputSize; i++)
        {
            outputGradients[i] = (outputOutputs[i] - labels[i]) / labels.length;
        }
        dot(outputGradients, transpose(outputWeights), hiddenGradients);
        reluGradient(hiddenOutputs, hiddenGradients);

        dot(transpose(hiddenOutputs), outputGradients, outputWeights, -learningRate);
        dot(transpose(inputs), hiddenGradients, hiddenWeights, -learningRate);
    *//*}
    public double[] predict(double[] inputs, int numClasses)
    {
        double[] hiddenOutputs = new double[hiddenSize];
        double[] outputOutputs = new double[numClasses];
        dot(inputs, hiddenWeights, hiddenOutputs);
        relu(hiddenOutputs);
        dot(hiddenOutputs, outputWeights, outputOutputs);
        softmax(outputOutputs);
        return outputOutputs;
    }
    private void reluGradient(double[] a, double[] b)
    {
        for(int i = 0; i < a.length; i++)
        {
            if(a[i] <= 0)
            {
                b[i] = 0;
            }
        }
    }
    private double[] transpose(double[] a)
    {
        double[] b = new double[a.length];
        int numRows = a .length / outputSize;
        int numCols = outputSize;
        for(int i = 0; i < numRows; i++)
        {
            for(int j = 0; j < numCols; j++)
            {
                b[j * numRows + 1] = a[i * numCols + j];
            }
        }
        return b;
    }
    private void dot(double[] a, double[] b, double[] c)
    {
        for(int i = 0; i < c.length; i++)
        {
            double sum = 0;
            for(int j = 0; j < a.length; j++)
            {
                sum += a[j] * b[j * c.length + i];
            }
            c[i] = sum;
        }
    }
    private void relu(double[] a)
    {
        for(int i = 0; i < a.length; i++)
        {
            if(a[i] < 0)
            {
                a[i] = 0;
            }
        }
    }
    private void softmax(double[] a)
    {
        double max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < a.length; i++)
        {
            if(a[i] > max)
            {
                max = a[i];
            }
        }
        double sum = 0;
        for(int i = 0; i < a.length; i++)
        {
            a[i] = Math.exp(a[i] - max);
            sum += a[i];
        }
        for(int i = 0; i < a.length; i++)
        {
            a[i] /= sum;
        }
    }
    private void dot(double[] a, double[] b, double[] c, double scalar)
    {
        for(int i = 0; i < c.length; i++)
        {
            double sum = 0;
            for(int j = 0; j < a.length; j++)
            {
                sum += a[j] * b[j * c.length + i];
            }
            c[i] -= scalar * sum;
        }
    }*/
}
