package project.digits.repo;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import project.digits.NeuralNetwork1;
import project.digits.NeuralNetworkDigits;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class RepositoryDigits {
    private static final String readTrainImages ="D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\train-images.idx3-ubyte";
    private static final String readTrainLabels ="D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\train-labels.idx1-ubyte";
    private static final String readTestImages = "D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\t10k-images.idx3-ubyte";
    private static final String readTestLabels = "D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\images\\digits\\t10k-labels.idx1-ubyte";
    private double[][] trainImages;
    private double[][] trainLabels;
    private double[][] testImages;
    private double[][] testLabels;
    private final static int numHiddenNodes = 250;
    private final static int batchSize = 64;
    private final static int epochs = 10;
    public RepositoryDigits() throws IOException {
        loadData();
        //trainData();
        trainDataWithoutTool();
    }
    private void trainDataWithoutTool()
    {
        //NeuralNetworkDigits model = new NeuralNetworkDigits(trainImages[0].length, 128, 10);
        //model.train(trainImages, trainLabels, 10, 0.01);
        //NeuralNetwork1 model = new NeuralNetwork1(0.01, trainImages.length, trainImages[0].length, trainImages, trainLabels, 10, 10, 100);
        NeuralNetwork1 model = new NeuralNetwork1(0.01, 4, 2, new double[][]{{0,0},{0,1},{1,0},{1,1}}, new double[][]{{0},{1},{1},{0}}, 400,1,4000);
        model.train();
        int nCorrect = 0;
/*        for(int i = 0; i < testImages.length; i++)
        {
            int prediction = model.predict(testImages[0]);
            if(prediction==testLabels[i])
            {
                nCorrect++;
            }
        }
        double accuracy = (double)nCorrect/testImages.length;
        System.out.println("Accuracy: " + accuracy);*/
    }
    private void loadData() throws IOException {
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
    }
    private void trainData() throws IOException {
        DenseLayer layer1 = new DenseLayer
                .Builder()
                .nIn(28*28)
                .nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build();
        OutputLayer layer2 = new OutputLayer
                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(numHiddenNodes)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build();
        MultiLayerConfiguration configuration = new NeuralNetConfiguration
                .Builder()
                .updater(new Nesterovs(0.01, 0.9))
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(layer1)
                .layer(layer2)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);
        for(int i = 0; i < epochs; i++)
        {
            model.fit(trainData);
            System.out.println("Epoch: " + i);
        }
        Evaluation evaluation = new Evaluation(10);
        while(testData.hasNext()){
            DataSet ds = testData.next();
            INDArray output = model.output(ds.getFeatures(), false);
            evaluation.eval(ds.getLabels(), output);
        }
        System.out.println(evaluation.confusionMatrix());
        System.out.println(evaluation.accuracy());
    }
}
