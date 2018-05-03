package main.transferfunctions;

/**
 *
 * @author b2198
 */
public class SimpleSigmoidFunction implements TransferFunction {
    
    private double scale;
    private double offset;
    
    public SimpleSigmoidFunction(double scale, double offset){
        this.scale = scale;
        this.offset = offset;
    }

    @Override
    public double f(double x) {
        return (double)(scale/(1+Math.expm1(-x)+1)+offset);
    }

    @Override
    public double dfdx(double x) {
        final double etotheminusx = Math.expm1(-x)+1;
        return scale*etotheminusx/((1+etotheminusx)*(1+etotheminusx));
    }
    
}
