using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Signum Activation Function
  /// </summary>
  public class SignActivation : IFunction
  {
    public SignActivation(double zeroValue = 1)
    {
      ZeroValue = zeroValue;
    }

    public readonly double ZeroValue;
    public string ID { get { return "SIGN"; } }
    public string Name { get { return "Signum"; } }

    public double Value(double r)
    {
      if (r==0) return ZeroValue;
      return (r<0) ? -1 : 1;
    }

    public double Derivative(double r)
    {
      return 0;
    }
  }
}
