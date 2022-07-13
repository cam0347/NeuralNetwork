package ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

@SuppressWarnings("ALL")
public class NeuralNetwork extends SupervisedNetwork {
    private final NNActivation[] activations;
    private final NNError error;

    public NeuralNetwork(double[][] data, double[][] objectives, NNParameters p) {
        this.data = data;
        this.objectives = objectives;
        this.weights = new double[p.getLayerSize().length][][];
        this.bias = new double[p.getLayerSize().length][];
        this.alpha = p.getAlpha();
        this.epochs = p.getEpochs();
        this.rescaled = false;
        this.rescalingLog = 0;
        this.activations = p.getActivations();
        this.error = p.getError();

        //initialize neurons and synapses
        for (int i = 0; i < p.getLayerSize().length; i++) {
            int prevLayerSize = i == 0 ? p.getInputSize() : p.getLayerSize()[i - 1];
            this.weights[i] = new double[p.getLayerSize()[i]][prevLayerSize];
            this.bias[i] = new double[p.getLayerSize()[i]];

            for (int j = 0; j < this.weights[i].length; j++) {
                for (int k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] = Math.random() * 2 - 1; //ranging from -1 to 1
                }
            }

            for (int j = 0; j < this.bias[i].length; j++) {
                this.bias[i][j] = Math.random() * 2 - 1; //ranging from -1 to 1
            }
        }

        /*if (p.getAutomaticRescaling()) {
            this.checkMagnitude();
        }*/

        super.minMaxNorm();
    }

    class NNEpochsPerformer extends Thread {
        private double weights[][][];
        private double bias[][];
        private final int outputLayerIndex;
        private final int epochs;

        public NNEpochsPerformer(double[][][] weights, double[][] bias, int epochs) {
            this.weights = weights;
            this.bias = bias;
            this.outputLayerIndex = weights.length - 1;
            this.epochs = epochs;
        }

        @Override
        public void run() {
            for (int e = 0; e < this.epochs; e++) {
                for (int k = 0; k < data.length; k++) {
                    double[][] values = new double[this.weights.length][]; //neurons activation
                    double[][] nets = new double[values.length][]; //neurons value (wx + b)
                    double[][] xDerivatives = new double[values.length][]; //x derivatives for the backpropagation

                    for (int l = 0; l < values.length; l++) { //for each layer
                        values[l] = new double[this.weights[l].length];
                        nets[l] = new double[this.weights[l].length];

                        for (int i = 0; i < values[l].length; i++) { //for each neuron of layer l
                            double[] input = l == 0 ? data[k] : values[l - 1];

                            for (int j = 0; j < this.weights[l][i].length; j++) { //for each weight of neuron i of layer l
                                values[l][i] += this.weights[l][i][j] * input[j];
                            }

                            values[l][i] += this.bias[l][i];
                            nets[l][i] = values[l][i];
                            values[l][i] = activation(activations[l], values[l][i]);
                        }
                    }

                    for (int l = outputLayerIndex; l >= 0; l--) { //for each layer
                        xDerivatives[l] = new double[l == 0 ? data[k].length : values[l - 1].length];

                        for (int n = 0; n < this.weights[l].length; n++) { //for each neuron of layer l
                            double dActivation = l == outputLayerIndex ? errorDerivative(error, objectives[k][n], values[outputLayerIndex][n]) : xDerivatives[l + 1][n];
                            double dInput = activationDerivative(activations[l], values[l][n], nets[l][n]);
                            double localDerivative = dActivation * dInput;

                            for (int i = 0; i < this.weights[l][n].length; i++) { //for each synapse of neuron n of layer l
                                xDerivatives[l][i] += localDerivative * this.weights[l][n][i];
                                double dWeight = l == 0 ? data[k][i] : values[l - 1][i];
                                this.weights[l][n][i] -= alpha * localDerivative * dWeight;
                                this.bias[l][n] -= alpha * localDerivative;
                            }
                        }
                    }
                }
            }
        }

        public double[][][] getWeights() {
            return this.weights;
        }

        public double[][] getBias() {
            return this.bias;
        }
    }

    @Override
    public void train() {
        final int nProcessors = Runtime.getRuntime().availableProcessors(); //use all virtual processors available
        final int nThreads = this.epochs >= nProcessors ? nProcessors : 1;
        System.out.println("Workload distributed on " + nThreads + " CPU(s)");

        NNEpochsPerformer[] ep = new NNEpochsPerformer[nThreads];
        double[][][] learnedWeights = new double[this.weights.length][][];
        double[][] learnedBias = new double[this.bias.length][];

        for (int i = 0; i < this.weights.length; i++) {
            learnedWeights[i] = new double[this.weights[i].length][];
            learnedBias[i] = new double[this.bias[i].length];

            for (int j = 0; j < this.weights[i].length; j++) {
                learnedWeights[i][j] = new double[this.weights[i][j].length];
            }
        }

        for (int i = 0; i < ep.length; i++) {
            ep[i] = new NNEpochsPerformer(this.weights.clone(), this.bias.clone(), (int) Math.ceil(this.epochs / nThreads));
        }

        System.out.println("Training...");
        long start = System.nanoTime();

        for (NNEpochsPerformer p: ep) {
            p.start();
        }

        int randIndex = (int)(Math.random() * this.data.length);
        double[] testData = this.data[randIndex];
        double[] testObj = this.objectives[randIndex];
        double totalAccuracy = 0.00;

        try {
            for (NNEpochsPerformer p: ep) {
                p.join();
                double[][][] w = p.getWeights();
                double[][] b = p.getBias();

                double acc = 0.00;
                this.weights = w;
                this.bias = b;
                double[] y = this.getAnswer(testData);
                for (int i = 0; i < testObj.length; i++) {
                    acc += this.error(this.error, testObj[i], y[i]);
                }

                acc /= testObj.length;
                acc *= 10;
                totalAccuracy += acc;

                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        for (int k = 0; k < w[i][j].length; k++) {
                            learnedWeights[i][j][k] += w[i][j][k] * acc;
                        }
                    }
                }

                for (int i = 0; i < b.length; i++) {
                    for (int j = 0; j < b[i].length; j++) {
                        learnedBias[i][j] += b[i][j] * acc;
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("NNEpochsPerformer join raised an error: " + e.getMessage());
        }

        for (int i = 0; i < learnedWeights.length; i++) {
            for (int j = 0; j < learnedWeights[i].length; j++) {
                for (int k = 0; k < learnedWeights[i][j].length; k++) {
                    learnedWeights[i][j][k] /= totalAccuracy;
                }
            }
        }

        for (int i = 0; i < learnedBias.length; i++) {
            for (int j = 0; j < learnedBias[i].length; j++) {
                learnedBias[i][j] /= totalAccuracy;
            }
        }

        if (this.rescaled) {
            int lastIndex = learnedBias.length - 1;

            for (int i = 0; i < learnedBias[lastIndex].length; i++) {
                learnedBias[lastIndex][i] *= Math.pow(10, this.rescalingLog);
            }
        }

        this.weights = learnedWeights.clone();
        this.bias = learnedBias.clone();

        super.printElapsedTime((System.nanoTime() - start) / 1000000);
    }

    @Override
    public void test(double[][] testSet, double[][] objectives) {
        double accuracy = 0.00;

        for (int i = 0; i < testSet.length; i++) {
            double[] data = testSet[i];
            double[] y = objectives[i];
            double[] out = this.getAnswer(data);

            for (int j = 0; j < y.length; j++) {
                accuracy += Math.exp(-Math.pow(y[j] - out[j], 2)) / y.length;
            }
        }

        accuracy /= testSet.length;
        accuracy *= 100;
        System.out.println("Model accuracy: " + accuracy + "%");
    }

    @Override
    public double[] getAnswer(double[] input) {
        if (input.length != this.data[0].length) {
            System.out.println("Prediction data length mismatch");
            return null;
        }

        double[][] values = new double[this.weights.length][];

        for (int l = 0; l < this.weights.length; l++) { //for each layer
            values[l] = new double[this.weights[l].length];
            double[] _input = l == 0 ? input : values[l - 1];

            for (int i = 0; i < this.weights[l].length; i++) { //for each neuron of layer l
                for (int j = 0; j < this.weights[l][i].length; j++) { //for each weight of neuron i of layer l
                    values[l][i] += this.weights[l][i][j] * _input[j];
                }

                values[l][i] += this.bias[l][i];
                values[l][i] = this.activation(this.activations[l], values[l][i]);
            }
        }

        return values[values.length - 1];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Neural Network\n");
        sb.append("training samples: ").append(this.data.length).append("\n");

        long nweights = 0, nneurons = 0;
        for (double[][] weight: this.weights) {
            nneurons += weight.length;

            for (double[] doubles: weight) {
                nweights += doubles.length;
            }
        }

        sb.append("neurons: ").append(nneurons).append("\n");
        sb.append("synapses: ").append(nweights).append("\n");
        sb.append("learning rate: ").append(this.alpha).append("\n");
        sb.append("epochs: ").append(this.epochs).append("\n");
        return sb.toString();
    }
}
