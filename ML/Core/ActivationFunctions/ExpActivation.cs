using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Exponent Activation Function
  /// </summary>
  public class ExpActivation : IActivationFunction
  {
    public string Name { get { return "EXP"; } }


    public double Value(double r)
    {
      return Math.Exp(r);
    }

    public double Derivative(double r)
    {
      return Math.Exp(r);
    }

    public double DerivativeFromValue(double y)
    {
      return y;
    }
  }
}
