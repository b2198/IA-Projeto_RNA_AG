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
        int[] hiddenAmount = new int[]{3,3};
        double[][] expectedOutput = new double[][]{
            {0, 0, 0, 0, 1},
            {1, 0, 0, 1, 0},
            {1, 0, 1, 0, 0},
            {0, 1, 0, 0, 0}
        };
        TransferFunction function;
        function = new SimpleSigmoidFunction(1.1,-0.05);
        //function = new SimpleHyperbolicTangentFunction();
        double learningRate = 0.4;
        double momentum = 0.7;
        MLPArtificialNeuralNetwork xor = new MLPArtificialNeuralNetwork(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        xor.initializeRandomWeights();
        
        System.out.println("XOR created, now training");
        xor.train(10000000,0.001);
        
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
