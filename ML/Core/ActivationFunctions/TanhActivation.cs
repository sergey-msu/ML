using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Hyperbolic Tangent Activation Function
  /// </summary>
  public class TanhActivation : IActivationFunction
  {
    public string ID { get { return "TANH"; } }
    public string Name { get { return "Hyperbolic Tangent"; } }

    public double Value(double r)
    {
      return 2.0D / (1.0D + Math.Exp(-2.0D*r)) - 1.0D;
    }

    public double Derivative(double r)
    {
      var val = 2.0D / (1.0D + Math.Exp(-2.0D*r)) - 1.0D;
      return 1 - val*val;
    }

    public double DerivativeFromValue(double y)
    {
      return 1.0D - y*y;
    }
  }
}
