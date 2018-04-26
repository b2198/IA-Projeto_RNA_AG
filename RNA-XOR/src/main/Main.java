package main;

import main.transferfunctions.SimpleSigmoidFunction;
import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class Main {
    public static void main(String[] args){
        double[][] input = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int hiddenAmount = 2;
        double[][] expectedOutput = new double[][]{
            {0},
            {1},
            {1},
            {0}
        };
        TransferFunction sigmoid = new SimpleSigmoidFunction();
        double learningRate = 0.001;
        SimpleNeuralNetwork xor = new SimpleNeuralNetwork(input, hiddenAmount, expectedOutput, sigmoid, learningRate);
        
        System.out.println("XOR created, now training");
        for(int i = 0; i < 100000; i++){
            xor.train();
        }
        System.out.println("training finished, results:");
        System.out.println("expected:\n"
                + "0 0 ->" + expectedOutput[0][0] + "\n"
                + "0 1 ->" + expectedOutput[1][0] + "\n"
                + "1 0 ->" + expectedOutput[2][0] + "\n"
                + "1 1 ->" + expectedOutput[3][0]);
        System.out.println("obtained:\n"
                + "0 0 ->" + xor.getOutput()[0][0] + "\n"
                + "0 1 ->" + xor.getOutput()[1][0] + "\n"
                + "1 0 ->" + xor.getOutput()[2][0] + "\n"
                + "1 1 ->" + xor.getOutput()[3][0]);
    }
}
