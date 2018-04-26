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
    
    public SimpleNeuralNetwork(double[][] input, int hiddenAmount, double[][] expectedOutput, TransferFunction function, double learningRate) {
        this.input = input;
        this.hidden = new double[input.length][hiddenAmount];
        this.output = new double[expectedOutput.length][expectedOutput[0].length];
        this.weights0 = new double[input[0].length][hiddenAmount];
        this.weights1 = new double[hiddenAmount][expectedOutput[0].length];
        this.expectedOutput = expectedOutput;
        this.function = function;
        this.learningRate = learningRate;
        
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
                + "weights0:\n" + MatrixOperations.toString(weights0) + "\n"
                + "weights1:\n" + MatrixOperations.toString(weights1) + "\n"
                + "expected output:\n" + MatrixOperations.toString(expectedOutput));
    }

    public double[][] getOutput() {
        return output;
    }
    
    public void feedForward(){
        //System.out.println("starting feed forward");
        
        //H = TF(I * W0)
        MatrixOperations.matrixMult(input, weights0, hidden);
        //System.out.println("I * W0 = \n" + MatrixOperations.toString(hidden));
        MatrixOperations.applyTransferFunction(function, hidden, hidden);
        //System.out.println("TF(I * W0) =\n" + MatrixOperations.toString(hidden));
        
        //O = TF(H * W1)
        MatrixOperations.matrixMult(hidden, weights1, output);
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
        //deltaH = LR * Eh *(elemento a elemento) dTF(H)/dx
        double[][] deltaH = MatrixOperations.transpose(weights1);
        //System.out.println("W1transposta =\n" + MatrixOperations.toString(deltaH));
        deltaH = MatrixOperations.matrixMult(deltaO, deltaH);
        //System.out.println("deltaO * W1transposta =\n" + MatrixOperations.toString(deltaH));
        
        MatrixOperations.elementByElementMult(learningRate, deltaH, deltaH);
        //System.out.println("LR * Eh =\n" + MatrixOperations.toString(deltaH));
        double[][] dTFHdx = MatrixOperations.applyTransferFunctionDerivative(function, hidden);
        //System.out.println("dTF(H)/dx =\n" + MatrixOperations.toString(dTFHdx));
        MatrixOperations.elementByElementMult(deltaH, dTFHdx, deltaH);
        //System.out.println("LR * Eh *(elemento a elemento) dTF(H)/dx =\n" + MatrixOperations.toString(deltaH));
        
        
        //Wo = Wo + Itransposta * deltaH
        double[][] itdeltaH = MatrixOperations.transpose(input);
        //System.out.println("Itransposta =\n" + MatrixOperations.toString(itdeltaH));
        itdeltaH = MatrixOperations.matrixMult(itdeltaH, deltaH);
        //System.out.println("Itransposta * deltaH =\n" + MatrixOperations.toString(itdeltaH));
        MatrixOperations.matrixAdd(weights0, itdeltaH, weights0);
        //System.out.println("W0 + Itransposta * deltaH =\n" + MatrixOperations.toString(weights0));
        
        //W1 = W1 + Htransposta * deltaO
        double[][] htdeltaO = MatrixOperations.transpose(hidden);
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
    
}
