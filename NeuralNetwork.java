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

        if (p.getAutomaticRescaling()) {
            this.checkRescaling();
        }

        /*System.out.println("Normalizing...");
        double[] dmax = new double[this.data[0].length], dmin = new double[this.data[0].length];
        double[] omax = new double[this.objectives[0].length], omin = new double[this.objectives[0].length];
        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                if (this.data[i][j] > dmax[j]) {
                    dmax[j] = this.data[i][j];
                }

                if (this.data[i][j] < dmin[j]) {
                    dmin[j] = this.data[i][j];
                }
            }

            for (int j = 0; j < this.objectives[i].length; j++) {
                if (this.objectives[i][j] > omax[j]) {
                    omax[j] = this.objectives[i][j];
                }

                if (this.objectives[i][j] < omin[j]) {
                    omin[j] = this.objectives[i][j];
                }
            }
        }

        double[] ddiff = new double[dmax.length], odiff = new double[omax.length];

        for (int i = 0; i < dmax.length; i++) {
            ddiff[i] = dmax[i] - dmin[i];
        }

        for (int i = 0; i < omax.length; i++) {
            odiff[i] = omax[i] - omin[i];
        }

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] = (2 * this.data[i][j] - dmax[j] - dmin[j]) / ddiff[j];
            }

            for (int j = 0; j < this.objectives[i].length; j++) {
                this.objectives[i][j] = (2 * this.objectives[i][j] - omax[j] - omin[j]) / odiff[j];
            }
        }*/
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

        try {
            for (NNEpochsPerformer p: ep) {
                p.join();

                double[][][] w = p.getWeights();
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        for (int k = 0; k < w[i][j].length; k++) {
                            learnedWeights[i][j][k] += w[i][j][k] / nThreads;
                        }
                    }
                }

                double[][] b = p.getBias();

                for (int i = 0; i < b.length; i++) {
                    for (int j = 0; j < b[i].length; j++) {
                        learnedBias[i][j] += b[i][j] / nThreads;
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("NNEpochsPerformer join raised an error: " + e.getMessage());
        }

        this.weights = learnedWeights.clone();
        this.bias = learnedBias.clone();

        this.printElapsedTime((System.nanoTime() - start) / 1000000);
    }

    class NNEpochsPerformer extends Thread {
        private double weights[][][];
        private double bias[][];
        private int outputLayerIndex;
        private int epochs;

        public NNEpochsPerformer(double[][][] weights, double[][] bias, int epochs) {
            this.weights = weights;
            this.bias = bias;
            this.outputLayerIndex = weights.length - 1;
            this.epochs = epochs;
        }

        @Override
        public void run() {
            System.out.println("Thread " + super.getId() + ", epochs assigned: " + this.epochs);

            for (int e = 0; e < this.epochs; e++) {
                for (int k = 0; k < data.length; k++) {
                    double[][] values = new double[this.weights.length][]; //neurons activation
                    double[][] nets = new double[values.length][]; //neurons value (wx + b)

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

                    double[][] xDerivatives = new double[values.length][]; //error derivatives with respect to x

                    /*dActivation: derivative of the error function with respect to the activation (neuron output)
                    dInput: dervative of the activation function with respect to the neuron's value (wx + b)
                    dWeight: derivative of the weighted sum with respect to the weight*/

                    /*
                    penso che il problema sia nella somma delle derivate di x di ciascun neurone nello strato l + 1,
                    come dimostrato nella regressione lineare a più output (con 1 neurone di output funziona, con più no).
                     */

                    for (int l = outputLayerIndex; l >= 0; l--) { //for each layer
                        xDerivatives[l] = new double[l == 0 ? data[k].length : values[l - 1].length];

                        System.out.println(objectives[k][0] + " " + values[outputLayerIndex][0]);
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

                            //System.out.println();
                        }

                        //System.out.println(xDerivatives[l][0]);
                    }
                }
            }

            /*if (rescaled) {
                for (int i = 0; i < this.weights[outputLayerIndex].length; i++) {
                    this.bias[outputLayerIndex][i] *= Math.pow(10, rescalingLog);
                }
            }*/
        }

        public double[][][] getWeights() {
            return this.weights;
        }

        public double[][] getBias() {
            return this.bias;
        }
    }

    private void printElapsedTime(long n) {
        long ms = n;
        int days = (int) Math.floor(ms / 86400000.0);
        ms -= days * 86400000.0;
        int hours = (int) Math.floor(ms / 3600000.0);
        ms -= hours * 3600000.0;
        int minutes = (int) Math.floor(ms / 60000.0);
        ms -= minutes * 60000.0;
        int seconds = (int) Math.floor(ms / 1000.0);
        ms -= seconds * 1000.0;

        System.out.print("Training complete [elapsed ");

        if (n > 86400000) {
            System.out.print(days + "d " + hours + "h " + minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 3600000) {
            System.out.print(hours + "h " + minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 60000) {
            System.out.print(minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 1000) {
            System.out.print(seconds + "s " + ms + "ms]");
        } else {
            System.out.print(ms + "ms]");
        }

        System.out.println();
    }

    private double activation(NNActivation activation, double x) {
        return switch (activation) {
            case SIGMOID -> 1.00 / (1.00 + Math.exp(-x));
            case TANH -> Math.tanh(x);
            case RELU -> x > 0 ? x : 0;
            case LINEAR -> x;
        };
    }

    private double errorDerivative(NNError f, double y, double out) {
        double d = 1 / this.weights[this.weights.length - 1].length;
        return d * switch(f) {
            case CROSS_ENTROPY -> -(y / out - (1 - y) / (1 - out));
            case MSE -> -2 * (y - out);
            case MAE -> (out - y) / Math.abs(y - out);
        };
    }

    private double activationDerivative(NNActivation f, double out, double net) {
        return switch(f) {
            case LINEAR -> 1;
            case SIGMOID -> out * (1 - out);
            case TANH -> 1 - Math.pow(Math.tanh(net), 2);
            case RELU -> net > 0 ? 1 : 0;
        };
    }

    @Override
    public double[] predict(double[] input) {
        if (input.length != this.data[0].length) {
            System.out.println("Prediction data length mismatch");
            return null;
        }

        if (this.rescaled) {
            for (int i = 0; i < input.length; i++) {
                input[i] /= Math.pow(10, this.rescalingLog);
            }
        }

        double[][] values = new double[this.weights.length][];

        for (int l = 0; l < this.weights.length; l++) { //for each layer
            values[l] = new double[this.weights[l].length];

            for (int i = 0; i < this.weights[l].length; i++) { //for each neuron of layer l
                for (int j = 0; j < this.weights[l][i].length; j++) { //for each weight of neuron i of layer l
                    values[l][i] += this.weights[l][i][j] * input[j];
                }

                values[l][i] += this.bias[l][i];
                values[l][i] = this.activation(this.activations[l], values[l][i]);
            }
        }

        if (this.rescaled) {
            for (int i = 0; i < values[values.length - 1].length; i++) {
                values[values.length - 1][i] *= Math.pow(10, this.rescalingLog);
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

    public String printWeights() {
        StringBuilder s = new StringBuilder();

        for (int l = 0; l < this.weights.length; l++) {
            s.append("Layer ").append(l).append(" (neurons: " + this.weights[l].length).append(")\n[\n");
            for (int n = 0; n < this.weights[l].length; n++) {
                s.append("[");
                for (double w: this.weights[l][n]) {
                    s.append(w).append(" ");
                }

                s.append("][b: " + this.bias[l][n] + "]\n");
            }

            s.append("]\n\n");
        }

        return s.toString();
    }

    public static ArrayList<double[]> getDataset(File f) {
        BufferedReader br = null;

        try {
            br = new BufferedReader(new FileReader(f));
        } catch (Exception e) {
            System.out.println("File not found: " + f.getAbsolutePath());
            return null;
        }

        try {
            ArrayList<double[]> ret = new ArrayList<double[]>();

            while(br.ready()) {
                String row = br.readLine();
                String[] data = row.split(";");
                double[] x = new double[data.length];
                for (int i = 0; i < x.length; i++) {
                    x[i] = Double.parseDouble(data[i]);
                }

                ret.add(x);
            }

            return ret;
        } catch (Exception e) {
            System.out.println("There was an error parsing the dataset: " + f.getAbsolutePath());
            return null;
        }
    }

    public static ArrayList<double[]> getDataset(String path) {
        return getDataset(new File(path));
    }

    public static double[][] getSamplesFromDataset(ArrayList<double[]> ds) {
        int length = ds.size();
        double[][] ret = new double[length][ds.get(0).length];

        for (int i = 0; i < length; i++) {
            double[] row = ds.get(i);

            for (int j = 0; j < ds.get(i).length - 1; j++) {
                ret[i][j] = ds.get(i)[j];
            }
        }

        return ret;
    }
}
