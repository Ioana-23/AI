package project.repo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import project.domain.ClassifierFLower;
import project.domain.Flower;
import project.domain.Pair;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class RepositoryFlowers {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_PURPLE = "\u001B[35m";
    private final String readFileFlower = "D:\\Laboratoare\\ANUL 2\\Semestrul 2\\InteligentaArtificiala\\Lab9\\project\\src\\main\\resources\\date\\flowers\\date1.txt";
    private List<Flower> flowers = new ArrayList<>();
    private List<ClassifierFLower> trainDataOutputFlower = new ArrayList<>();
    private List<ClassifierFLower> testDataOutputFlower = new ArrayList<>();
    private List<List<Double>> trainDataInputsFlower = new ArrayList<>();
    private List<List<Double>> testDataInputsFlower = new ArrayList<>();
    private static final int epochs=10000;
    private static final double learningRate = 0.01;
    public RepositoryFlowers()
    {
        loadData();
        splitData();
        //normalizeDataFlower();
        trainModel();
        System.out.println();
        normalizeDataFlower();
    }
    private void loadData()
    {
        try {
            File myObj = new File(readFileFlower);
            Scanner myReader = new Scanner(myObj);
            while(myReader.hasNextLine())
            {
                String text1 = myReader.nextLine();
                String[] wordsText1 = text1.split(",");
                double sl = Double.parseDouble(wordsText1[0]);
                double sw = Double.parseDouble(wordsText1[1]);
                double pl = Double.parseDouble(wordsText1[2]);
                double pw = Double.parseDouble(wordsText1[3]);
                String[] classifierTextWords = wordsText1[4].split("-");
                String classifier1 = classifierTextWords[0];
                String classifier2 = classifierTextWords[1].substring(0, 1).toUpperCase() + classifierTextWords[1].substring(1, classifierTextWords[1].length());
                ClassifierFLower classifier = ClassifierFLower.valueOf(classifier1 + classifier2);
                Flower flower = new Flower(sl, sw, pl, pw, classifier);
                flowers.add(flower);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
    private void splitData()
    {
        Random random = new Random();
        for(int i = 0; i < flowers.size(); i++)
        {
            float value = random.nextFloat();
            if(value<0.2)
            {
                testDataInputsFlower.add(new ArrayList<>(Arrays.asList(flowers.get(i).getSepalLenth(), flowers.get(i).getSepalWidth(),flowers.get(i).getPetalLength(), flowers.get(i).getPetalWidth())));
                testDataOutputFlower.add(flowers.get(i).getClassifierFLower());
            }
            else
            {
                trainDataInputsFlower.add(new ArrayList<>(Arrays.asList(flowers.get(i).getSepalLenth(), flowers.get(i).getSepalWidth(),flowers.get(i).getPetalLength(), flowers.get(i).getPetalWidth())));
                trainDataOutputFlower.add(flowers.get(i).getClassifierFLower());
            }
        }
    }
    private void normalizeDataFlower()
    {
        List<Double> valueList = new ArrayList<>();
        double sum = 0;
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            valueList.add(trainDataInputsFlower.get(i).get(0));
            sum+=trainDataInputsFlower.get(i).get(0);
        }
        double m = sum / valueList.size();
        double s2 = 0;
        for(int i = 0; i < valueList.size(); i++)
        {
            double s1 = Math.pow(valueList.get(i) - m, 2);
            s2+=s1;
        }
        double s = Math.sqrt((1.0f / (valueList.size()-1)) * s2);
        for(int i = 0; i < valueList.size(); i++)
        {
            double v1 = (valueList.get(i) - m) / s;
            trainDataInputsFlower.get(i).remove(0);
            trainDataInputsFlower.get(i).add(0, v1);
        }

        valueList.clear();
        sum = 0;
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            valueList.add(trainDataInputsFlower.get(i).get(1));
            sum+=trainDataInputsFlower.get(i).get(1);
        }
        m = sum / valueList.size();
        s2 = 0;
        for(int i = 0; i < valueList.size(); i++)
        {
            double s1 = Math.pow(valueList.get(i) - m, 2);
            s2+=s1;
        }
        s = Math.sqrt((1.0f / (valueList.size()-1)) * s2);
        for(int i = 0; i < valueList.size(); i++)
        {
            double v1 = (valueList.get(i) - m) / s;
            trainDataInputsFlower.get(i).remove(1);
            trainDataInputsFlower.get(i).add(1, v1);
        }

        valueList.clear();
        sum = 0;
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            valueList.add(trainDataInputsFlower.get(i).get(2));
            sum+=trainDataInputsFlower.get(i).get(2);
        }
        m = sum / valueList.size();
        s2 = 0;
        for(int i = 0; i < valueList.size(); i++)
        {
            double s1 = Math.pow(valueList.get(i) - m, 2);
            s2+=s1;
        }
        s = Math.sqrt((1.0f / (valueList.size()-1)) * s2);
        for(int i = 0; i < valueList.size(); i++)
        {
            double v1 = (valueList.get(i) - m) / s;
            trainDataInputsFlower.get(i).remove(2);
            trainDataInputsFlower.get(i).add(2, v1);
        }

        valueList.clear();
        sum = 0;
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            valueList.add(trainDataInputsFlower.get(i).get(3));
            sum+=trainDataInputsFlower.get(i).get(3);
        }
        m = sum / valueList.size();
        s2 = 0;
        for(int i = 0; i < valueList.size(); i++)
        {
            double s1 = Math.pow(valueList.get(i) - m, 2);
            s2+=s1;
        }
        s = Math.sqrt((1.0f / (valueList.size()-1)) * s2);
        for(int i = 0; i < valueList.size(); i++)
        {
            double v1 = (valueList.get(i) - m) / s;
            trainDataInputsFlower.get(i).remove(3);
            trainDataInputsFlower.get(i).add(3, v1);
        }
    }
    private void trainModel()
    {
        double[][] x = new double[trainDataInputsFlower.size()][4];
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            for(int j = 0; j < 4; j++)
            {
                x[i][j] = trainDataInputsFlower.get(i).get(j);
            }
        }
        double[] y = new double[trainDataInputsFlower.size()];
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            if(trainDataOutputFlower.get(i).equals(ClassifierFLower.IrisSetosa))
            {
                y[i] = 1;
            }
            else
            {
                y[i] = 0;
            }
        }
        OutputLayer layer2 = new OutputLayer
                .Builder()
                .nIn(4)
                .nOut(3)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build();
        MultiLayerConfiguration config = new NeuralNetConfiguration
                .Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(layer2)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        INDArray input = Nd4j.create(trainDataInputsFlower.size(), 4);
        INDArray label = Nd4j.create(trainDataInputsFlower.size(), 3);
        for(int i = 0; i < trainDataInputsFlower.size(); i++)
        {
            input.putScalar(i, 0, x[i][0]);
            input.putScalar(i, 1, x[i][1]);
            input.putScalar(i, 2, x[i][2]);
            input.putScalar(i, 3, x[i][3]);
            if(trainDataOutputFlower.get(i).equals(ClassifierFLower.IrisSetosa))
            {
                label.putScalar(i, 0, 1);
                label.putScalar(i, 1, 0);
                label.putScalar(i, 2, 0);
            }
            else if(trainDataOutputFlower.get(i).equals(ClassifierFLower.IrisVersicolor))
            {
                label.putScalar(i, 0, 0);
                label.putScalar(i, 1, 1);
                label.putScalar(i, 2, 0);
            }
            else
            {
                label.putScalar(i, 0, 0);
                label.putScalar(i, 1, 0);
                label.putScalar(i, 2, 1);
            }
        }
        for(int i =  0; i < epochs; i++)
        {
            model.fit(input, label);
        }

        INDArray inputTest = Nd4j.create(testDataInputsFlower.size(), 4);
        INDArray labelTest = Nd4j.create(testDataInputsFlower.size(), 3);
        for(int i = 0; i < testDataInputsFlower.size(); i++)
        {
            for(int j = 0; j < 4; j++)
            {
                inputTest.putScalar(i, j, testDataInputsFlower.get(i).get(j));
            }
        }
        for(int i = 0; i < testDataInputsFlower.size(); i++)
        {
            INDArray predictedOutputs = model.output(inputTest);
            labelTest.putScalar(i, 0, predictedOutputs.getDouble(i, 0));
            labelTest.putScalar(i, 1, predictedOutputs.getDouble(i, 1));
            labelTest.putScalar(i, 2, predictedOutputs.getDouble(i, 2));
        }
        INDArray realOutput = Nd4j.create(testDataInputsFlower.size(), 3);
        for(int i = 0; i < testDataInputsFlower.size(); i++)
        {
            if(testDataOutputFlower.get(i).equals(ClassifierFLower.IrisSetosa))
            {
                realOutput.putScalar(i, 0, 1);
                realOutput.putScalar(i, 1, 0);
                realOutput.putScalar(i, 2, 0);
            }
            else if(testDataOutputFlower.get(i).equals(ClassifierFLower.IrisVersicolor))
            {
                realOutput.putScalar(i, 0, 0);
                realOutput.putScalar(i, 1, 1);
                realOutput.putScalar(i, 2, 0);
            }
            else
            {
                realOutput.putScalar(i, 0, 0);
                realOutput.putScalar(i, 1, 0);
                realOutput.putScalar(i, 2, 1);
            }
        }
        DataSet dataSet = new DataSet(inputTest, realOutput);
        DataSetIterator dataSetIterator = new SamplingDataSetIterator(dataSet, testDataInputsFlower.size(), testDataInputsFlower.size());
        Evaluation prediction = model.evaluate(dataSetIterator);
        System.out.println(ANSI_PURPLE +  prediction.confusionMatrix() + ANSI_RESET);
        System.out.println(ANSI_YELLOW + "Precision: " + prediction.precision() + ANSI_RESET);
        System.out.println(ANSI_PURPLE + "Acuratetea: " + prediction.accuracy() + ANSI_RESET);
    }
}
