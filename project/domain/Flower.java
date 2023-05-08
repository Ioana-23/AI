package project.domain;

public class Flower {
    private double sepalLenth;
    private double sepalWidth;
    private double petalLength;
    private double petalWidth;
    private ClassifierFLower classifierFLower;

    public Flower(double sepalLenth, double sepalWidth, double petalLength, double petalWidth, ClassifierFLower classifierFLower) {
        this.sepalLenth = sepalLenth;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.classifierFLower = classifierFLower;
    }

    public double getSepalLenth() {
        return sepalLenth;
    }

    public double getSepalWidth() {
        return sepalWidth;
    }

    public double getPetalLength() {
        return petalLength;
    }

    public double getPetalWidth() {
        return petalWidth;
    }

    public ClassifierFLower getClassifierFLower() {
        return classifierFLower;
    }
}
