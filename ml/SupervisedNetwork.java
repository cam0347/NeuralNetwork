package ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

public abstract class SupervisedNetwork {
    protected double[][] data;
    protected double[][] objectives;
    protected double[][][] weights;
    protected double[][] bias;
    protected double alpha;
    protected int epochs;
    protected boolean rescaled;
    protected int rescalingLog;

    protected void checkMagnitude() {
        double max = this.getMinMaxAvg()[1];

        int log;
        if ((log = (int) Math.floor(Math.log10(max))) >= 1) {
            this.rescaled = true;
            this.rescalingLog = log;
            this.rescale(log);
        }
    }

    private double[] getMinMaxAvg() {
        double min = this.data[0][0];
        double max = min;
        double avg = 0.00;
        double dataLength = this.data.length * this.data[0].length;

        for (double[] v: this.data) {
            for (double d: v) {
                if (d > max) {
                    max = d;
                }

                if (d < min) {
                    min = d;
                }

                avg += d / dataLength;
            }
        }

        return new double[] {min, max, avg};
    }

    protected void minMaxNorm() {
        double[] minMax = this.getMinMaxAvg();
        double diff = minMax[1] - minMax[0];

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] = (this.data[i][j] - minMax[0]) / diff;
            }
        }
    }

    protected void zScoreNorm() {
        double mean = this.getMinMaxAvg()[2];
        double sigma = 0.00;

        for (double[] v : this.data) {
            for (double d: v) {
                sigma += (Math.pow(d - mean, 2)) / this.data.length;
            }
        }

        sigma = Math.sqrt(sigma);

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] = (this.data[i][j] - mean) / sigma;
            }
        }
    }

    protected void meanNorm() {
        double[] values = this.getMinMaxAvg();

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] = (this.data[i][j] - values[0]) / (values[1] - values[2]);
            }
        }
    }

    protected void rescale(int log) {
        log = (int) Math.pow(10, log);
        System.out.println("Rescaling features...");

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] /= log;
            }

            for (int j = 0; j < this.objectives[i].length; j++) {
                this.objectives[i][j] /= log;
            }
        }
    }

    protected double activation(NNActivation activation, double x) {
        return switch (activation) {
            case SIGMOID -> 1.00 / (1.00 + Math.exp(-x));
            case TANH -> Math.tanh(x);
            case RELU -> x > 0 ? x : 0;
            case LINEAR -> x;
        };
    }

    protected double error(NNError f, double y, double out) {
        return switch(f) {
            case MSE -> Math.pow(y - out, 2);
            case MAE -> Math.abs(y - out);
            case CROSS_ENTROPY -> -(y * Math.exp(out) + (1 - y) * Math.exp(1 - out));
        };
    }

    protected void printElapsedTime(long n) {
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

    protected double errorDerivative(NNError f, double y, double out) {
        double d = 1.00 / this.weights[this.weights.length - 1].length;

        d *= switch(f) {
            case CROSS_ENTROPY -> -(y / out - (1 - y) / (1 - out));
            case MSE -> -2 * (y - out);
            case MAE -> (out - y) / Math.abs(y - out);
        };

        if (Double.isNaN(d)) {
            System.out.println("NaN value found: " + f + ", y=" + y + ", out=" + out);
            System.exit(0);
        }

        return d;
    }

    protected double activationDerivative(NNActivation f, double out, double net) {
        return switch(f) {
            case LINEAR -> 1;
            case SIGMOID -> out * (1 - out);
            case TANH -> 1 - Math.pow(Math.tanh(net), 2);
            case RELU -> net > 0 ? 1 : 0;
        };
    }

    public String printWeights() {
        StringBuilder s = new StringBuilder();

        for (int l = 0; l < this.weights.length; l++) {
            s.append("Layer ").append(l).append(" (neurons: ").append(this.weights[l].length).append(")\n[\n");
            for (int n = 0; n < this.weights[l].length; n++) {
                s.append("[");
                for (double w: this.weights[l][n]) {
                    s.append(w).append(" ");
                }

                s.append("][b: ").append(this.bias[l][n]).append("]\n");
            }

            s.append("]\n\n");
        }

        return s.toString();
    }

    public static double[][][] loadCSV(String filename, int outputCols) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            ArrayList<double[]>[] list = new ArrayList[] {new ArrayList(), new ArrayList()};
            double[][][] ret = new double[2][][];

            while(br.ready()) {
                String row = br.readLine();

                String regex = row.contains(";") ? ";" : ",";
                String[] data = row.split(regex);

                double[] x = new double[data.length - outputCols];
                double[] y = new double[outputCols];

                for (int i = 0; i < x.length; i++) {
                    x[i] = Double.parseDouble(data[i]);
                }

                for (int i = 0; i < outputCols; i++) {
                    y[i] = Double.parseDouble(data[x.length + i]);
                }

                list[0].add(x);
                list[1].add(y);
            }

            ret[0] = new double[list[0].size()][list[0].get(0).length];
            ret[1] = new double[list[1].size()][outputCols];

            for (int i = 0; i < list[0].size(); i++) {
                ret[0][i] = list[0].get(i);
            }

            for (int i = 0; i < list[1].size(); i++) {
                ret[1][i] = list[1].get(i);
            }

            return ret;
        } catch (FileNotFoundException fnf) {
            System.out.println("Dataset not found");
        } catch (Exception e) {
            System.out.println("There was an error: " + e);
        }

        System.exit(0);
        return null;
    }

    public abstract void test(double[][] testSet, double[][] objectives);
    public abstract double[] getAnswer(double[] x);
    public abstract void train();
}
