package ml;

public abstract class SupervisedNetwork {
    protected double[][] data;
    protected double[][] objectives;
    protected double[][][] weights;
    protected double[][] bias;
    protected double alpha;
    protected int epochs;
    protected boolean rescaled;
    protected int rescalingLog;

    protected void checkRescaling() {
        System.out.println("Checking input magnitude...");

        double max = this.data[0][0];
        for (double[] d: this.data) {
            for (double v: d) {
                if (v > max) {
                    max = v;
                }
            }
        }

        int log;
        if ((log = (int) Math.floor(Math.log10(max))) >= 2) {
            this.rescaled = true;
            this.rescalingLog = log;
            this.rescale(log);
        }
    }

    protected void rescale(int log) {
        log = (int) Math.pow(10, log);
        System.out.println("Rescaling...");

        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                this.data[i][j] /= log;
            }

            for (int j = 0; j < this.objectives[i].length; j++) {
                this.objectives[i][j] /= log;
            }
        }
    }

    public void importModel(double[][][] weights, double[][] bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double[] predict(double[] x) {
        if (x.length != this.weights[0][0].length) {
            System.out.println("Error: input length mismatch");
            return new double[] {0};
        }

        double ret = 0.00;

        for (int i = 0; i < x.length; i++) {
            ret += this.weights[0][0][i] * x[i];
        }

        return new double[] {ret};
    }

    public abstract void train();
}
