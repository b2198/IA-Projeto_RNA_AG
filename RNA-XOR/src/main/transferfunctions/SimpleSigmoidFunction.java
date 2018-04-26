package main.transferfunctions;

/**
 *
 * @author b2198
 */
public class SimpleSigmoidFunction implements TransferFunction {

    @Override
    public double f(double x) {
        return (double)(1/(1+Math.expm1(-x)+1));
    }

    @Override
    public double dfdx(double x) {
        return f(x)*(1-f(x));
    }
    
}
