package test;
import ml.*;

@SuppressWarnings("ALL")
public class Main {
    public static void main(String[] args) {
        double[][] data = SupervisedNetwork.loadCSV("/users/camillozavattaro/desktop/dataset/autoencoder.csv", 4)[0];
        NeuralNetwork nn = new NeuralNetwork(data, data, new NNParameters(4, new int[] {3, 4}, 200, 0.0001, true, new NNActivation[] {NNActivation.LINEAR, NNActivation.LINEAR}, NNError.MAE));
        nn.train();

        nn.test(new double[][] {{500, 912, -7, -134}, {0, -1000, 12, 56}}, new double[][] {{500, 912, -7, -134}, {0, -1000, 12, 56}});

        for (double d: nn.getAnswer(new double[] {500, 912, -7, -134})) {
            System.out.println(d);
        }
    }
}
