package main;

import java.util.ArrayList;
import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class MLPArtificialNeuralNetwork {
    private double[][] input;
    private double[][][] hidden;
    private double[][] output;
    private double[][][] weights;
    private double[][] expectedOutput;
    private double[][][] errors;
    private TransferFunction function;
    private double learningRate;
    //input with bias
    private double[][] xi;
    //hidden with bias
    private double[][][] xh;
    private double momentum;
    private double[][][] lastUpdateVectors;
    private boolean[] lastUpdateVectorsInitialized;
    private ArrayList<MLPListener> listeners;
    
    public MLPArtificialNeuralNetwork(double[][] input, int[] hiddenAmount, double[][] expectedOutput, TransferFunction function, double learningRate, double momentum) {
        set(input, hiddenAmount, expectedOutput, function, learningRate, momentum, true);
        listeners = new ArrayList<>();
    }
    
    public void set(double[][]input, int[] hiddenAmounts, double[][] expectedOutput, TransferFunction function, double learningRate, double momentum, boolean createWeights){
        if(input.length != expectedOutput.length){
            throw new RuntimeException("Input and expected output lengths don't match");
        }
        this.input = input;
        this.hidden = new double[hiddenAmounts.length][input.length][];
        for(int i = 0; i < hiddenAmounts.length; i++){
            for(int j = 0; j < input.length; j++){
                this.hidden[i][j] = new double[hiddenAmounts[i]];
            }
        }
        this.output = new double[input.length][expectedOutput[0].length];
        this.expectedOutput = expectedOutput;
        if(createWeights){
            createWeights(hiddenAmounts);
        }
        this.function = function;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.errors = new double[hiddenAmounts.length+1][][];
        for(int i = 0; i < hiddenAmounts.length; i++){
            this.errors[i] = new double[input.length][hiddenAmounts[i]+1];
        }
        this.errors[hiddenAmounts.length] = new double[input.length][expectedOutput[0].length];
        
        
        xi = new double[input.length][input[0].length+1];
        setXi();
        xh = new double[hiddenAmounts.length][][];
        for(int i = 0; i < hiddenAmounts.length; i++){
            xh[i] = new double[input.length][hiddenAmounts[i]+1];
            setXh(i);
        }
        
        lastUpdateVectors = new double[hiddenAmounts.length+1][][];
        lastUpdateVectorsInitialized = new boolean[hiddenAmounts.length+1];
        for(int i = 0; i < hiddenAmounts.length+1; i++){
            lastUpdateVectors[i] = new double[weights[i].length][];
            for(int j = 0; j < weights[i].length; j++){
                lastUpdateVectors[i][j] = new double[weights[i][j].length];
            }
        }
    }
    
    public ArrayList<MLPListener> getListeners(){
        return listeners;
    }
    
    private void createWeights(int[] hiddenAmounts){
        this.weights = new double[hiddenAmounts.length+1][][];
        this.weights[0] = new double[input[0].length+1][hiddenAmounts[0]];
        for(int i = 1; i < hiddenAmounts.length; i++){
            this.weights[i] = new double[hiddenAmounts[i-1]+1][hiddenAmounts[i]];
        }
        this.weights[hiddenAmounts.length] = new double[hiddenAmounts[hiddenAmounts.length-1]+1][expectedOutput[0].length];
    }
    
    public void initializeRandomWeights(){
        //initialize weights with random values
        for(int i = 0; i < weights.length; i++){
            for(int j = 0; j < weights[i].length; j++){
                for(int k = 0; k < weights[i][j].length; k++){
                    weights[i][j][k] = Math.random();
                }
            }
        }
    }

    public double[][] getOutput() {
        return output;
    }
    
    public void feedForward(){
        //H0 = TF(Xi * W0)
        setXi();
        MatrixOperations.matrixMult(xi, weights[0], hidden[0]);
        MatrixOperations.applyTransferFunction(function, hidden[0], hidden[0]);
        
        //Hn = TF(Xhn-1 * Wn)
        for(int i = 1; i < hidden.length; i++){
            setXh(i-1);
            MatrixOperations.matrixMult(xh[i-1], weights[i], hidden[i]);
            MatrixOperations.applyTransferFunction(function, hidden[i], hidden[i]);
        }
        
        //O = TF(Xhlength-1 * Wlength)
        setXh(hidden.length-1);
        MatrixOperations.matrixMult(xh[hidden.length-1], weights[hidden.length], output);
        MatrixOperations.applyTransferFunction(function, output, output);
        
    }
    
    public void backPropagation(){
        
        //Eo = Oesperado - O;
        //deltaO = LR * Eo *(elemento a elemento) dTF(O)/dx
        MatrixOperations.matrixSub(expectedOutput, output, errors[hidden.length]);
        
        double[][] deltaO = MatrixOperations.applyTransferFunctionDerivative(function, output);
        MatrixOperations.elementByElementMult(errors[hidden.length], deltaO, deltaO);
        MatrixOperations.elementByElementMult(learningRate, deltaO, deltaO);
        
        //Ehlength-1 = deltaO * Wlengthtransposta
        //deltaHlength-1 = LR * Ehlength-1 *(elemento a elemento) dTF(Xhlength-1)/dx
        double[][] wlengtht = MatrixOperations.transpose(weights[hidden.length]);
        MatrixOperations.matrixMult(deltaO, wlengtht, errors[hidden.length-1]);
        
        double[][] deltaHlengthm1 = MatrixOperations.applyTransferFunctionDerivative(function, xh[hidden.length-1]);
        MatrixOperations.elementByElementMult(errors[hidden.length-1], deltaHlengthm1, deltaHlengthm1);
        MatrixOperations.elementByElementMult(learningRate, deltaHlengthm1, deltaHlengthm1);
        
        //Wlength = Wlength + Xhlength-1transposta * deltaO
        double[][] xhlengthm1t = MatrixOperations.transpose(xh[hidden.length-1]);
        double[][] xhlengthm1tdeltaO = MatrixOperations.matrixMult(xhlengthm1t, deltaO);
        //momentum application
        if(!lastUpdateVectorsInitialized[hidden.length]){
            lastUpdateVectorsInitialized[hidden.length] = true;
        } else {
            MatrixOperations.elementByElementMult(momentum, lastUpdateVectors[hidden.length], lastUpdateVectors[hidden.length]);
            MatrixOperations.matrixAdd(lastUpdateVectors[hidden.length], xhlengthm1tdeltaO, xhlengthm1tdeltaO);
        }
        MatrixOperations.copyTo(xhlengthm1tdeltaO, lastUpdateVectors[hidden.length]);
        //end of momentum application
        MatrixOperations.matrixAdd(weights[hidden.length], xhlengthm1tdeltaO, weights[hidden.length]);
        
        
        double[][] yLastDelta = new double[deltaHlengthm1.length][deltaHlengthm1[0].length-1];
        for(int i = 0; i < yLastDelta.length; i++){
            for(int j = 0; j < yLastDelta[i].length; j++){
                yLastDelta[i][j] = deltaHlengthm1[i][j];
            }
        }
        for(int i = hidden.length-2; i >= 0; i--){
            //Ehn = YdeltaHn+1 * Wn+1transposta
            //deltaHn = LR * Ehn *(elemento a elemento) dTF(Xhn-1)/dx
            double[][] wnp1t = MatrixOperations.transpose(weights[i+1]);
            MatrixOperations.matrixMult(yLastDelta, wnp1t, errors[i]);
            
            double[][] deltaHn = MatrixOperations.applyTransferFunctionDerivative(function, xh[i]);
            MatrixOperations.elementByElementMult(errors[i], deltaHn, deltaHn);
            MatrixOperations.elementByElementMult(learningRate, deltaHn, deltaHn);
            
            //Wn+1 = Wn+1 + Xhntransposta * YdeltaHn+1
            double[][] xhnt = MatrixOperations.transpose(xh[i]);
            double[][] xhntYdeltahnp1 = MatrixOperations.matrixMult(xhnt, yLastDelta);
            //momentum application
            if(!lastUpdateVectorsInitialized[i+1]){
                lastUpdateVectorsInitialized[i+1] = true;
            } else {
                MatrixOperations.elementByElementMult(momentum, lastUpdateVectors[i+1], lastUpdateVectors[i+1]);
                MatrixOperations.matrixAdd(lastUpdateVectors[i+1], xhntYdeltahnp1, xhntYdeltahnp1);
            }
            MatrixOperations.copyTo(xhntYdeltahnp1, lastUpdateVectors[i+1]);
            //end of momentum application
            MatrixOperations.matrixAdd(weights[i+1], xhntYdeltahnp1, weights[i+1]);
            
            yLastDelta = new double[deltaHn.length][deltaHn[0].length-1];
            for(int j = 0; j < yLastDelta.length; j++){
                for(int k = 0; k < yLastDelta[j].length; k++){
                    yLastDelta[j][k] = deltaHn[j][k];
                }
            }
        }
        
        
        //W0 = W0 + Xitransposta * YdeltaH0
        double[][] xit = MatrixOperations.transpose(xi);
        double[][] xitXdeltaH = MatrixOperations.matrixMult(xit, yLastDelta);
        //momentum application
        if(!lastUpdateVectorsInitialized[0]){
            lastUpdateVectorsInitialized[0] = true;
        } else {
            MatrixOperations.elementByElementMult(momentum, lastUpdateVectors[0], lastUpdateVectors[0]);
            MatrixOperations.matrixAdd(lastUpdateVectors[0], xitXdeltaH, xitXdeltaH);
        }
        MatrixOperations.copyTo(xitXdeltaH, lastUpdateVectors[0]);
        //end of momentum application
        MatrixOperations.matrixAdd(weights[0], xitXdeltaH, weights[0]);
        
    }
    
    public void train(int maxEpochNumber, double maxAverageErrorAccepted){
        MLPInfo info = new MLPInfo(input, hidden, output, weights, xi, xh, errors, lastUpdateVectors, function, learningRate);
        for(MLPListener listener : listeners){
            listener.onTrainingStarted(info);
        }
        long time = System.nanoTime();
        boolean success = false;
        double averageError = 0;
        int epoch = -1;
        for(int i = 0; i < maxEpochNumber; i++){
            for(MLPListener listener : listeners){
                listener.onCycleStart(info, i);
            }
            feedForward();
            backPropagation();
            averageError = 0;
            for(int j = 0; j < errors[errors.length-1].length; j++){
                for(int k = 0; k < errors[errors.length-1][j].length; k++){
                    averageError += Math.abs(errors[errors.length-1][j][k]);
                }
            }
            for(MLPListener listener : listeners){
                listener.onCycleEnd(info, i, averageError);
            }
            if(averageError <= maxAverageErrorAccepted){
                success = true;
                epoch = i;
                break;
            }
        }
        time = System.nanoTime()-time;
        for(MLPListener listener : listeners){
            listener.onFinished(info, success?epoch:maxEpochNumber, success, averageError);
        }
        if(success){
            System.out.println("Training successful");
            System.out.println("error obtained: " + averageError);
            System.out.println("epochs needed: " + epoch);
            System.out.printf("time spent:%.2f seconds\n",(time/1e9));
            System.out.println("output:\n" + MatrixOperations.toString(output));
        } else {
            System.out.println("Training failed");
            System.out.println("error obtained: " + averageError);
            System.out.println("epochs needed: " + maxEpochNumber);
            System.out.printf("time spent:%.2f seconds\n",(time/1e9));
            System.out.println("output:\n" + MatrixOperations.toString(output));
        }
    }
    
    public void test(double[][] testInput){
        if(!MatrixOperations.isMatrix(testInput)){
            throw new RuntimeException("the input is not a matrix");
        }
        if(testInput[0].length != input[0].length){
            throw new RuntimeException("the input does not have the correct amount of columns");
        }
        double[][] tempInput = input;
        setInput(testInput);
        feedForward();
        System.out.println("starting test:");
        System.out.println("output:\n" + MatrixOperations.toString(output));
        setInput(tempInput);
    }
    
    public void setInput(double[][] input){
        int[] hiddenAmounts = new int[hidden.length];
        for(int i = 0; i < hiddenAmounts.length; i++){
            hiddenAmounts[i] = hidden[i][0].length;
        }
        set(input,hiddenAmounts,new double[input.length][output[0].length],function,learningRate,momentum, false);
    }
    
    private void setXi(){
        for(int i = 0; i < xi.length; i++){
            for(int j = 0; j <xi[i].length; j++){
                if(j < xi[i].length-1){
                    xi[i][j] = input[i][j];
                } else {
                    xi[i][j] = 1;
                }
            }
        }
    }
    
    private void setXh(int i){
        for(int j = 0; j < xh[i].length; j++){
            for(int k = 0; k <xh[i][j].length-1; k++){
                xh[i][j][k] = hidden[i][j][k];
            }
            xh[i][j][xh[i][j].length-1] = 1;
        }
    }
    
    public interface MLPListener{
        void onTrainingStarted(MLPInfo info);
        void onCycleStart(MLPInfo info, int cycle);
        void onCycleEnd(MLPInfo info, int cycle, double averageError);
        void onFinished(MLPInfo info, int finalEpoch, boolean success, double finalError);
    }
    
    public class MLPInfo{
        public final double[][] input;
        public final double[][][] hidden;
        public final double[][] output;
        public final double[][][] weights;
        public final double[][] xi;
        public final double[][][] xh;
        public final double[][][] errors;
        public final double[][][] lastUpdateVectors;
        public final TransferFunction function;
        public final double learningRate;

        public MLPInfo(double[][] input, double[][][] hidden, double[][] output, double[][][] weights, double[][] xi, double[][][] xh, double[][][] errors, double[][][] lastUpdateVectors, TransferFunction function, double learningRate) {
            this.input = input;
            this.hidden = hidden;
            this.output = output;
            this.weights = weights;
            this.xi = xi;
            this.xh = xh;
            this.errors = errors;
            this.lastUpdateVectors = lastUpdateVectors;
            this.function = function;
            this.learningRate = learningRate;
        }
        
    }
}
