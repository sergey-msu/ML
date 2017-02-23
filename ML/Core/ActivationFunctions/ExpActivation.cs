using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Exponent Activation Function
  /// </summary>
  public class ExpActivation : IFunction
  {
    public string ID { get { return "EXP"; } }
    public string Name { get { return "Exponent"; } }

    public double Value(double r)
    {
      return Math.Exp(r);
    }

    public double Derivative(double r)
    {
      return Math.Exp(r);
    }
  }
}
