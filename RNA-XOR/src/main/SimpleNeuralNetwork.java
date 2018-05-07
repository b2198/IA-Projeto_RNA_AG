package main;

import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class SimpleNeuralNetwork {
    private double[][] input;
    private double[][] hidden;
    private double[][] output;
    private double[][] weights0;
    private double[][] weights1;
    private double[][] expectedOutput;
    private double[][] oError;
    private double[][] hError;
    private TransferFunction function;
    private double learningRate;
    //input with bias
    private double[][] xi;
    //hidden with bias
    private double[][] xh;
    private double momentum;
    private double[][] lastUpdateVector0;
    private double[][] lastUpdateVector1;
    private boolean lastUpdateVector0Initialized;
    private boolean lastUpdateVector1Initialized;
    
    public SimpleNeuralNetwork(double[][] input, int hiddenAmount, double[][] expectedOutput, TransferFunction function, double learningRate, double momentum) {
        set(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        System.out.println("Simple Neural Network created, initial state:\n"
                + "input:\n" + MatrixOperations.toString(input) + "\n"
                + "hidden:\n" + MatrixOperations.toString(hidden) + "\n"
                + "output:\n" + MatrixOperations.toString(output) + "\n"
                + "Xi:\n" + MatrixOperations.toString(xi) + "\n"
                + "Xh:\n" + MatrixOperations.toString(xh) + "\n"
                + "weights0:\n" + MatrixOperations.toString(weights0) + "\n"
                + "weights1:\n" + MatrixOperations.toString(weights1) + "\n"
                + "expected output:\n" + MatrixOperations.toString(expectedOutput));
    }
    
    public void set(double[][]input, int hiddenAmount, double[][] expectedOutput, TransferFunction function, double learningRate, double momentum){
        if(input.length != expectedOutput.length){
            throw new RuntimeException("Input and expected output lengths don't match");
        }
        this.input = input;
        this.hidden = new double[input.length][hiddenAmount];
        this.output = new double[input.length][expectedOutput[0].length];
        this.weights0 = new double[input[0].length+1][hiddenAmount];
        this.weights1 = new double[hiddenAmount+1][expectedOutput[0].length];
        this.expectedOutput = expectedOutput;
        this.function = function;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.oError = new double[input.length][expectedOutput[0].length];
        this.hError = new double[input.length][weights1.length];
        
        xi = new double[input.length][input[0].length+1];
        xh = new double[input.length][hiddenAmount+1];
        
        setXi();
        setXh();
        
        //initialize weights0 with random values
        for(int i = 0; i < weights0.length; i++){
            for(int j = 0; j < weights0[i].length; j++){
                weights0[i][j] = Math.random();
            }
        }
        //initialize weights1 with random values
        for(int i = 0; i < weights1.length; i++){
            for(int j = 0; j < weights1[i].length; j++){
                weights1[i][j] = (double)Math.random();
            }
        }
        
        
        lastUpdateVector0 = new double[weights0.length][weights0[0].length];
        lastUpdateVector1 = new double[weights1.length][weights1[0].length];
    }

    public double[][] getOutput() {
        return output;
    }
    
    public void feedForward(){
        //H = TF(Xi * W0)
        setXi();
        MatrixOperations.matrixMult(xi, weights0, hidden);
        MatrixOperations.applyTransferFunction(function, hidden, hidden);
        
        //O = TF(Xh * W1)
        setXh();
        MatrixOperations.matrixMult(xh, weights1, output);
        MatrixOperations.applyTransferFunction(function, output, output);
        
    }
    
    public void backPropagation(){
        
        //Eo = Oesperado - O;
        //deltaO = LR * Eo *(elemento a elemento) dTF(O)/dx
        MatrixOperations.matrixSub(expectedOutput, output, oError);
        
        double[][] deltaO = MatrixOperations.applyTransferFunctionDerivative(function, output);
        MatrixOperations.elementByElementMult(oError, deltaO, deltaO);
        MatrixOperations.elementByElementMult(learningRate, deltaO, deltaO);
        
        
        //Eh = deltaO * W1transposta
        //deltaH = LR * Eh *(elemento a elemento) dTF(Xh)/dx
        double[][] w1t = MatrixOperations.transpose(weights1);
        MatrixOperations.matrixMult(deltaO, w1t, hError);
        
        double[][] deltaH = MatrixOperations.applyTransferFunctionDerivative(function, xh);
        MatrixOperations.elementByElementMult(hError, deltaH, deltaH);
        MatrixOperations.elementByElementMult(learningRate, deltaH, deltaH);
        
        
        //W0 = W0 + Xitransposta * XdeltaH
        double[][] xit = MatrixOperations.transpose(xi);
        double[][] xDeltaH = new double[deltaH.length][deltaH[0].length-1];
        for(int i = 0; i < xDeltaH.length; i++){
            for(int j = 0; j < xDeltaH[i].length; j++){
                xDeltaH[i][j] = deltaH[i][j];
            }
        }
        double[][] xitXdeltaH = MatrixOperations.matrixMult(xit, xDeltaH);
        //momentum application
        if(!lastUpdateVector0Initialized){
            lastUpdateVector0Initialized = true;
        } else {
            MatrixOperations.elementByElementMult(momentum, lastUpdateVector0, lastUpdateVector0);
            MatrixOperations.matrixAdd(lastUpdateVector0, xitXdeltaH, xitXdeltaH);
        }
        MatrixOperations.copyTo(xitXdeltaH, lastUpdateVector0);
        //end of momentum application
        MatrixOperations.matrixAdd(weights0, xitXdeltaH, weights0);
        
        //W1 = W1 + Xhtransposta * deltaO
        double[][] xht = MatrixOperations.transpose(xh);
        double[][] xhtdeltaO = MatrixOperations.matrixMult(xht, deltaO);
        //momentum application
        if(!lastUpdateVector1Initialized){
            lastUpdateVector1Initialized = true;
        } else {
            MatrixOperations.elementByElementMult(momentum, lastUpdateVector1, lastUpdateVector1);
            MatrixOperations.matrixAdd(lastUpdateVector1, xhtdeltaO, xhtdeltaO);
        }
        MatrixOperations.copyTo(xhtdeltaO, lastUpdateVector1);
        //end of momentum application
        MatrixOperations.matrixAdd(weights1, xhtdeltaO, weights1);
    }
    
    public void train(){
        feedForward();
        backPropagation();
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
    
    private void setXh(){
        for(int i = 0; i < xh.length; i++){
            for(int j = 0; j <xh[i].length; j++){
                if(j < xh[i].length-1){
                    xh[i][j] = hidden[i][j];
                } else {
                    xh[i][j] = 1;
                }
            }
        }
    }
    
}
