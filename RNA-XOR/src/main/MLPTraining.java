package main;

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.scene.Scene;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import main.graphics.MLPPlotter;
import main.graphics.EpochErrorMLPPlotter;
import main.transferfunctions.SimpleSigmoidFunction;
import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class MLPTraining extends Application {
    
    public static void main(String[] args){
        launch(args);
    }
    
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        
        MLPPlotter timeError = new EpochErrorMLPPlotter(0.02,10);
        MLPPlotter timeError2 = new EpochErrorMLPPlotter(0.005,10);
        HBox hbox = new HBox(timeError,timeError2);
        VBox vbox = new VBox(hbox);
        StackPane root = new StackPane(vbox);
        Scene scene = new Scene(root);
        primaryStage.setScene(scene);
        primaryStage.show();
        
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
        double learningRate = 0.04;
        double momentum = 0.8;
        MLPArtificialNeuralNetwork xor = new MLPArtificialNeuralNetwork(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        xor.initializeRandomWeights();
        xor.getListeners().add(timeError);
        xor.getListeners().add(timeError2);
        
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
        
        System.out.println("XOR created, now training");
        Task trainTask = new Task(){
            @Override
            protected String call() throws Exception {
                xor.train(10000000,0.001);
                return "trained";
            }

            @Override
            protected void succeeded() {
                xor.test(testInput);
                xor.test(testInput2);
            }

            @Override
            protected void failed() {
                System.out.println("something has gone wrong");
            }
            
            
        };
        Thread thread = new Thread(trainTask);
        thread.setDaemon(true);
        thread.start();
        
        
        
    }
    
}
