package main;

import main.transferfunctions.SimpleHyperbolicTangentFunction;
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
        int[] hiddenAmount = new int[]{3,2};
        double[][] expectedOutput = new double[][]{
            {0},
            {1},
            {1},
            {0}
        };
        TransferFunction function;
        function = new SimpleSigmoidFunction(1.05,-0.025);
        //function = new SimpleHyperbolicTangentFunction();
        double learningRate = 0.3;
        double momentum = 0.7;
        SimpleNeuralNetwork xor = new SimpleNeuralNetwork(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        xor.initializeRandomWeights();
        
        System.out.println("XOR created, now training");
        xor.train(5000000,0.001);
        
        double[][] testInput = new double[][]{
            {0,0}
        };
        
        double[][] testInput2 = new double[][]{
            {0,0},
            {0,1},
            {1,0},
            {1,1},
            {1,0},
            {0,1},
            {1,0}
        };
        
        xor.test(testInput);
        xor.test(testInput2);
    }
}
