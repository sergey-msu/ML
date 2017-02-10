using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Signum Activation Function
  /// </summary>
  public class SignActivation : IFunction
  {
    public string ID { get { return "SIGN"; } }
    public string Name { get { return "Signum"; } }

    public double Value(double r)
    {
      if (r==0) return 0;
      return (r<0) ? -1 : 1;
    }

    public double Derivative(double r)
    {
      return 0;
    }
  }
}
