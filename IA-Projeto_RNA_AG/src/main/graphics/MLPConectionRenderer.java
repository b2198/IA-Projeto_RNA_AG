package main.graphics;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.util.Duration;
import main.MLPArtificialNeuralNetwork;
import main.MatrixOperations;

/**
 *
 * @author b2198
 */
public class MLPConectionRenderer extends MLPRenderer{
    
    private int inputAmount;
    private int[] hiddenAmount;
    private int outputAmount;
    private double[][][] weights;
    private double currentError;
    
    private Timeline loop;
    
    private double inputRatio;
    private double[] hiddenRatio;
    private double outputRatio;
    private double horizontalRatio;
    private double auxInputRatio;
    private double[] auxHiddenRatio;
    private double auxOutputRatio;
    
    public MLPConectionRenderer(){
        loop = new Timeline(60, new KeyFrame(Duration.ONE, (e)->render(), (KeyValue)null));
        loop.setCycleCount(Animation.INDEFINITE);
    }
    
    private void render(){
        GraphicsContext gc = this.getGraphicsContext2D();
        
        gc.setFill(Color.WHITE);
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(1);
        gc.fillRect(0, 0, getWidth(), getHeight());
        gc.strokeRect(0, 0, getWidth(), getHeight());
        
        gc.strokeText("current error: "+String.format("%.6f", currentError), 300, 20, 90);
        
        //draw inputs
        for(int i = 0; i < inputAmount; i++){
            gc.strokeOval(
                    2*horizontalRatio-auxInputRatio,
                    (2+3*i)*inputRatio-auxInputRatio,
                    2*auxInputRatio,
                    2*auxInputRatio
            );
        }
        //draw input bias
        gc.setFill(Color.BLACK);
        gc.fillOval(
                2*horizontalRatio-auxInputRatio,
                400-2*auxInputRatio-auxInputRatio,
                2*auxInputRatio,
                2*auxInputRatio
        );
        
        //draw hiddens
        for(int i = 0; i < hiddenAmount.length; i++){
            //draw hidden
            for(int j = 0; j < hiddenAmount[i]; j++){
                gc.strokeOval(
                        (2+3*(i+1))*horizontalRatio-auxHiddenRatio[i],
                        (2+3*j)*hiddenRatio[i]-auxHiddenRatio[i],
                        2*auxHiddenRatio[i],
                        2*auxHiddenRatio[i]
                );
            }
            
            //draw hidden bias
            gc.fillOval(
                    (2+3*(i+1))*horizontalRatio-auxHiddenRatio[i],
                    400-2*auxHiddenRatio[i]-auxHiddenRatio[i],
                    2*auxHiddenRatio[i],
                    2*auxHiddenRatio[i]
            );
        }
        
        //draw output
        for(int i = 0; i < outputAmount; i++){
            gc.strokeOval(
                    (2+3*(1+hiddenAmount.length))*horizontalRatio-auxOutputRatio,
                    (2+3*i)*outputRatio-auxOutputRatio,
                    2*auxOutputRatio,
                    2*auxOutputRatio
            );
        }
        
        final double COLOR_MIDDLE = 0.5;
        
        //draw input to hidden weights
        for(int i = 0; i < inputAmount+1; i++){
            for(int j = 0; j < hiddenAmount[0]; j++){
                gc.setLineWidth(1);
                double change = Math.abs(weights[0][i][j]);
                change = 1*change/(COLOR_MIDDLE+change);
                gc.setStroke(Color.color(
                        (weights[0][i][j] >= 0 ? change : 0),
                        0,
                        (weights[0][i][j] < 0 ? change : 0)
                ));
                if(i == inputAmount){
                    gc.strokeLine(
                            2*horizontalRatio+auxInputRatio,
                            400-2*auxInputRatio,
                            (2+3*1)*horizontalRatio-auxHiddenRatio[0],
                            (2+3*j)*hiddenRatio[0]
                    );
                } else {
                    gc.strokeLine(
                            2*horizontalRatio+auxInputRatio,
                            (2+3*i)*inputRatio,
                            (2+3*1)*horizontalRatio-auxHiddenRatio[0],
                            (2+3*j)*hiddenRatio[0]
                    );
                }
            }
        }
        
        //draw hidden to hidden weights
        for(int i = 0; i < hiddenAmount.length-1; i++){
            //draw i hidden to i+1 hidden weights
            for(int j = 0; j < hiddenAmount[i]+1; j++){
                for(int k = 0; k < hiddenAmount[i+1]; k++){
                    gc.setLineWidth(1);
                    double change = Math.abs(weights[i+1][j][k]);
                    change = 1*change/(COLOR_MIDDLE+change);
                    gc.setStroke(Color.color(
                            (weights[i+1][j][k] >= 0 ? change : 0),
                            0,
                            (weights[i+1][j][k] < 0 ? change : 0)
                    ));
                    if(j == hiddenAmount[i]){
                        gc.strokeLine(
                                (2+3*(i+1))*horizontalRatio+auxHiddenRatio[i],
                                400-2*auxHiddenRatio[i],
                                (2+3*(i+2))*horizontalRatio-auxHiddenRatio[i+1],
                                (2+3*k)*hiddenRatio[i+1]
                        );
                    } else {
                        gc.strokeLine(
                                (2+3*(i+1))*horizontalRatio+auxHiddenRatio[i],
                                (2+3*j)*hiddenRatio[i],
                                (2+3*(i+2))*horizontalRatio-auxHiddenRatio[i+1],
                                (2+3*k)*hiddenRatio[i+1]
                        );
                    }
                }
            }
        }
        
        //draw hidden to output weights
        for(int i = 0; i < hiddenAmount[hiddenAmount.length-1]+1; i++){
            for(int j = 0; j < outputAmount; j++){
                gc.setLineWidth(1);
                double change = Math.abs(weights[hiddenAmount.length][i][j]);
                change = 1*change/(COLOR_MIDDLE+change);
                gc.setStroke(Color.color(
                        (weights[hiddenAmount.length][i][j] >= 0 ? change : 0),
                        0,
                        (weights[hiddenAmount.length][i][j] < 0 ? change : 0)
                ));
                if(i == hiddenAmount[hiddenAmount.length-1]){
                    gc.strokeLine(
                            (2+3*(hiddenAmount.length-1+1))*horizontalRatio+auxHiddenRatio[hiddenAmount.length-1],
                            400-2*auxHiddenRatio[hiddenAmount.length-1],
                            (2+3*(1+hiddenAmount.length))*horizontalRatio-auxOutputRatio,
                            (2+3*j)*outputRatio
                    );
                } else {
                    gc.strokeLine(
                            (2+3*(hiddenAmount.length-1+1))*horizontalRatio+auxHiddenRatio[hiddenAmount.length-1],
                            (2+3*i)*hiddenRatio[hiddenAmount.length-1],
                            (2+3*(1+hiddenAmount.length))*horizontalRatio-auxOutputRatio,
                            (2+3*j)*outputRatio
                    );
                }
            }
        }
    }

    @Override
    public void onTrainingStarted(MLPArtificialNeuralNetwork.MLPInfo info) {
        inputAmount = info.input[0].length;
        inputRatio = 400.0/(3*(inputAmount+1)+1);
        hiddenAmount = new int[info.hidden.length];
        hiddenRatio = new double[hiddenAmount.length];
        for(int i = 0; i < hiddenAmount.length; i++){
            hiddenAmount[i] = info.hidden[i][0].length;
            hiddenRatio[i] = 400.0/(3*(hiddenAmount[i]+1)+1);
        }
        outputAmount = info.output[0].length;
        outputRatio = 400.0/(3*outputAmount+1);
        
        horizontalRatio = 400.0/(3*(1+hiddenAmount.length+1)+1);
        auxInputRatio = Math.min(inputRatio, horizontalRatio);
        auxHiddenRatio = new double[hiddenRatio.length];
        for(int i = 0; i < hiddenAmount.length; i++){
            auxHiddenRatio[i] = Math.min(hiddenRatio[i], horizontalRatio);
        }
        auxOutputRatio = Math.min(outputRatio, horizontalRatio);
        
        weights = new double[info.weights.length][][];
        for(int i = 0; i < weights.length; i++){
            weights[i] = new double[info.weights[i].length][info.weights[i][0].length];
            MatrixOperations.copyTo(info.weights[i], weights[i]);
        }
        
        loop.play();
        try{
            Thread.sleep(100);
        } catch(InterruptedException ex){
            System.out.println("thread exception: " + ex);
        }
    }

    @Override
    public void onCycleStart(MLPArtificialNeuralNetwork.MLPInfo info, int cycle) {
    }

    @Override
    public void onCycleEnd(MLPArtificialNeuralNetwork.MLPInfo info, int cycle, double averageError) {
        for(int i = 0; i < weights.length; i++){
            MatrixOperations.copyTo(info.weights[i], weights[i]);
        }
        currentError = averageError;
    }

    @Override
    public void onFinished(MLPArtificialNeuralNetwork.MLPInfo info, int finalEpoch, boolean success, double finalError) {
        loop.stop();
        currentError = finalError;
        render();
    }
    
}
