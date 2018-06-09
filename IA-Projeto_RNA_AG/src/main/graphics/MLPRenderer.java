package main.graphics;

import javafx.scene.canvas.Canvas;
import main.MLPArtificialNeuralNetwork.MLPListener;

/**
 *
 * @author b2198
 */
public abstract class MLPRenderer extends Canvas implements MLPListener {
    
    public MLPRenderer(){
        super(400,400);
    }
    
}
