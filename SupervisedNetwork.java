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

        /*for (int i = 0; i < this.data[0].length; i++) {
            double max = this.data[0][0];

            for (double[] x: this.data) {
                if (x[i] > max) {
                    max = x[i];
                }
            }

            int log;
            if ((log = (int) Math.floor(Math.log10(max))) >= 1) {
                log += 2;
                this.rescaled = true;
                this.rescalingLog = log;
                this.rescale(log);
            }
        }*/
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
    
    /*public boolean importModel(String file) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String row = br.readLine();
            String[] data = row.split(";");
            
            if (data.length <= 1) {
                return false;
            }
            
            this.weights = new double[1][1][data.length - 1];
            this.bias = Double.parseDouble(data[0]);
            
            for (int i = 0; i < data.length - 1; i++) {
                this.weights[0][0][i] = Double.parseDouble(data[i + 1]);
            }
            
            return false;
        } catch (Exception e) {
            return false;
        }
    }
    
    public boolean exportModel(String path) {
        try {
            FileWriter fw = new FileWriter(path);
            fw.write(this.bias + ";");

            for (int i = 0; i < this.weights[0][0].length - 1; i++) {
                fw.write(this.weights[0][0][i] + ";");
            }

            fw.write(this.weights[0][0][this.weights[0][0].length - 1] + "");
            fw.flush();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean exportModel() {
        File dir = new File(".");
        String path = dir.getAbsolutePath();
        return exportModel(path + "/SupervisedModel.csv");
    }*/

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
