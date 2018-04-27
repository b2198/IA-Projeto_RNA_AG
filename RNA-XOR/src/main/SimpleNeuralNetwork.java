package main;

import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class SimpleNeuralNetwork {
    private final double[][] input;
    private final double[][] hidden;
    private final double[][] output;
    private final double[][] weights0;
    private final double[][] weights1;
    private final double[][] expectedOutput;
    private final TransferFunction function;
    private final double learningRate;
    private final double[][] Xi;
    private final double[][] Xh;
    
    public SimpleNeuralNetwork(double[][] input, int hiddenAmount, double[][] expectedOutput, TransferFunction function, double learningRate) {
        this.input = input;
        this.hidden = new double[input.length][hiddenAmount];
        this.output = new double[expectedOutput.length][expectedOutput[0].length];
        this.weights0 = new double[input[0].length+1][hiddenAmount];
        this.weights1 = new double[hiddenAmount+1][expectedOutput[0].length];
        this.expectedOutput = expectedOutput;
        this.function = function;
        this.learningRate = learningRate;
        
        Xi = new double[input.length][input[0].length+1];
        Xh = new double[input.length][hiddenAmount+1];
        
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

        System.out.println("Simple Neural Network created, initial state:\n"
                + "input:\n" + MatrixOperations.toString(input) + "\n"
                + "hidden:\n" + MatrixOperations.toString(hidden) + "\n"
                + "output:\n" + MatrixOperations.toString(output) + "\n"
                + "Xi:\n" + MatrixOperations.toString(Xi) + "\n"
                + "Xh:\n" + MatrixOperations.toString(Xh) + "\n"
                + "weights0:\n" + MatrixOperations.toString(weights0) + "\n"
                + "weights1:\n" + MatrixOperations.toString(weights1) + "\n"
                + "expected output:\n" + MatrixOperations.toString(expectedOutput));
    }

    public double[][] getOutput() {
        return output;
    }
    
    public void feedForward(){
        //System.out.println("starting feed forward");
        
        //H = TF(Xi * W0)
        setXi();
        MatrixOperations.matrixMult(Xi, weights0, hidden);
        //System.out.println("I * W0 = \n" + MatrixOperations.toString(hidden));
        MatrixOperations.applyTransferFunction(function, hidden, hidden);
        //System.out.println("TF(I * W0) =\n" + MatrixOperations.toString(hidden));
        
        //O = TF(Xh * W1)
        setXh();
        MatrixOperations.matrixMult(Xh, weights1, output);
        //System.out.println("H * W1 =\n" + MatrixOperations.toString(output));
        MatrixOperations.applyTransferFunction(function, output, output);
        //System.out.println("TF(H * W1) =\n" + MatrixOperations.toString(output));
        
        //System.out.println("feed forward finished");
    }
    
    public void backPropagation(){
        //System.out.println("starting back propagation");
        
        //Eo = Oesperado - O;
        //deltaO = LR * Eo *(elemento a elemento) dTF(O)/dx
        double[][] deltaO = MatrixOperations.matrixSub(expectedOutput, output);
        //System.out.println("Oesperado - O =\n" + MatrixOperations.toString(deltaO));
        
        MatrixOperations.elementByElementMult(learningRate, deltaO, deltaO);
        //System.out.println("LR * Eo =\n" + MatrixOperations.toString(deltaO));
        double[][] dTFOdx = MatrixOperations.applyTransferFunctionDerivative(function, output);
        //System.out.println("dTF(O)/dx =\n" + MatrixOperations.toString(dTFOdx));
        MatrixOperations.elementByElementMult(deltaO, dTFOdx, deltaO);
        //System.out.println("LR * Eo *(elemento a elemento) dTF(O)/dx =\n" + MatrixOperations.toString(deltaO));
        
        
        //Eh = deltaO * W1transposta
        //deltaH = LR * Eh *(elemento a elemento) dTF(Xh)/dx
        double[][] deltaH = MatrixOperations.transpose(weights1);
        //System.out.println("W1transposta =\n" + MatrixOperations.toString(deltaH));
        deltaH = MatrixOperations.matrixMult(deltaO, deltaH);
        //System.out.println("deltaO * W1transposta =\n" + MatrixOperations.toString(deltaH));
        
        MatrixOperations.elementByElementMult(learningRate, deltaH, deltaH);
        //System.out.println("LR * Eh =\n" + MatrixOperations.toString(deltaH));
        double[][] dTFXhdx = MatrixOperations.applyTransferFunctionDerivative(function, Xh);
        //System.out.println("dTF(H)/dx =\n" + MatrixOperations.toString(dTFHdx));
        MatrixOperations.elementByElementMult(deltaH, dTFXhdx, deltaH);
        //System.out.println("LR * Eh *(elemento a elemento) dTF(H)/dx =\n" + MatrixOperations.toString(deltaH));
        
        
        //W0 = W0 + Xitransposta * XdeltaH
        setXi();
        double[][] xitXdeltaH = MatrixOperations.transpose(Xi);
        double[][] xDeltaH = new double[deltaH.length][deltaH[0].length-1];
        for(int i = 0; i < xDeltaH.length; i++){
            for(int j = 0; j < xDeltaH[i].length; j++){
                xDeltaH[i][j] = deltaH[i][j];
            }
        }
        //System.out.println("Itransposta =\n" + MatrixOperations.toString(itdeltaH));
        xitXdeltaH = MatrixOperations.matrixMult(xitXdeltaH, xDeltaH);
        //System.out.println("Itransposta * deltaH =\n" + MatrixOperations.toString(itdeltaH));
        MatrixOperations.matrixAdd(weights0, xitXdeltaH, weights0);
        //System.out.println("W0 + Itransposta * deltaH =\n" + MatrixOperations.toString(weights0));
        
        //W1 = W1 + Xhtransposta * deltaO
        setXh();
        double[][] htdeltaO = MatrixOperations.transpose(Xh);
        //System.out.println("Htransposta =\n" + MatrixOperations.toString(htdeltaO));
        htdeltaO = MatrixOperations.matrixMult(htdeltaO, deltaO);
        //System.out.println("Htransposta * deltaO =\n" + MatrixOperations.toString(htdeltaO));
        MatrixOperations.matrixAdd(weights1, htdeltaO, weights1);
        //System.out.println("W1 + Htransposta * deltaO =\n" + MatrixOperations.toString(weights1));
        
        //System.out.println("back propagation finished");
    }
    
    public void train(){
        feedForward();
        backPropagation();
    }
    
    private void setXi(){
        for(int i = 0; i < Xi.length; i++){
            for(int j = 0; j <Xi[i].length; j++){
                if(j < Xi[i].length-1){
                    Xi[i][j] = input[i][j];
                } else {
                    Xi[i][j] = 1;
                }
            }
        }
    }
    
    private void setXh(){
        for(int i = 0; i < Xh.length; i++){
            for(int j = 0; j <Xh[i].length; j++){
                if(j < Xh[i].length-1){
                    Xh[i][j] = hidden[i][j];
                } else {
                    Xh[i][j] = 1;
                }
            }
        }
    }
    
}
