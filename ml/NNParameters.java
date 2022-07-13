package ml;

public class NNParameters {
    private final int inputSize;
    private final int[] layerSize;
    private final int epochs;
    private final double alpha;
    private final boolean automaticRescaling;
    private final NNActivation[] activations;
    private final NNError error;

    public NNParameters(int inputSize, int[] layerSize, int epochs, double alpha, boolean automaticRescaling, NNActivation[] activations, NNError error) {
        if (layerSize.length != activations.length) {
            throw new IllegalArgumentException("The number of layers must be equal to the number of activations");
        }

        if (inputSize < 0 || epochs < 0) {
            throw new IllegalArgumentException("The input size and epochs must be positive");
        }

        if (alpha < 0) {
            System.out.println("[NEGATIVE LEARNING RATE]");
        }

        this.inputSize = inputSize;
        this.layerSize = layerSize;
        this.epochs = epochs;
        this.alpha = alpha;
        this.automaticRescaling = automaticRescaling;
        this.activations = activations;
        this.error = error;
    }

    public NNParameters() {
        this.inputSize = 1;
        this.layerSize = new int[] {2};
        this.epochs = 10000;
        this.alpha = 0.0001;
        this.automaticRescaling = true;
        this.activations = new NNActivation[] {NNActivation.LINEAR};
        this.error = NNError.MSE;
    }

    public int getInputSize() {
        return this.inputSize;
    }

    public int[] getLayerSize() {
        return this.layerSize;
    }

    public int getEpochs() {
        return this.epochs;
    }

    public double getAlpha() {
        return this.alpha;
    }

    public boolean getAutomaticRescaling() {
        return this.automaticRescaling;
    }

    public NNActivation[] getActivations() {
        return this.activations;
    }

    public NNError getError() {
        return this.error;
    }
}
